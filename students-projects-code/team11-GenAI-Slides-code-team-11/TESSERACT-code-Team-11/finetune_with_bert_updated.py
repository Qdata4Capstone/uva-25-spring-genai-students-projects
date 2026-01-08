import os
import random
import re
import argparse
from glob import glob
from pathlib import Path
from typing import Tuple, Dict, Any

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from segmentation_models_pytorch import Unet
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# ──────────────────────────────  Local imports  ──────────────────────────── #
from data_loader import OASISDataset  # noqa: E402 – your own dataset class


# ───────────────────────────────  Utilities  ─────────────────────────────── #
def get_log_file(log_path):
    """Get a file handle for logging."""
    return open(log_path, "w", encoding="utf‑8")


def log(msg: str, log_file, *, end: str = "\n") -> None:
    """Print *and* append to log file (flushes immediately)."""
    print(msg, end=end)
    log_file.write(f"{msg}{end}")
    log_file.flush()


def set_seed(seed: int = 0) -> None:
    """Make numpy / Python / PyTorch deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dice_coeff_and_loss(
    pred, target, smooth=1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return *(dice_coeff, dice_loss)* for binary masks."""
    pred = (torch.sigmoid(pred) > 0.5).float()
    
    # Make tensors contiguous and flatten them
    pred = pred.contiguous().reshape(-1)
    target = target.contiguous().reshape(-1)
    
    # Calculate intersection and union
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice, 1 - dice  # Return Dice Coefficient and Dice Loss


def reshape_tensor(t: torch.Tensor) -> torch.Tensor:
    """[B, 160, 192, 224] → swap dims 1&2 |→ [B×192, 1, 160, 224]."""
    b, d1, d2, d3 = t.shape
    return t.permute(0, 2, 1, 3).reshape(b * d2, 1, d1, d3)


def save_prediction_vs_mask_gif(
    images: torch.Tensor,
    logits: torch.Tensor,
    masks: torch.Tensor,
    *,
    save_path: str,
    caption: str = "",
    fps: int = 5,
) -> None:
    """Save a GIF with *Image | Prediction | Ground‑truth* panels.

    The *caption* (typically the reference text) is centred above each frame.
    """

    images = images.squeeze(1).cpu().numpy()  # [T, H, W]
    preds = (torch.sigmoid(logits) > 0.5).float().squeeze(1).cpu().numpy()
    masks = masks.squeeze(1).cpu().numpy()

    frames = []
    for t in range(images.shape[0]):
        # Create a figure with minimal spacing and no padding
        fig = plt.figure(figsize=(7.5, 2.5), constrained_layout=False)
        
        # Create grid with no spacing between panels
        gs = fig.add_gridspec(nrows=1, ncols=3, wspace=0, hspace=0)
        
        # Add subplots
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        
        # Plot images with no padding
        ax1.imshow(images[t], cmap="gray")
        ax1.axis("off")
        ax1.text(0.5, -0.05, "Image", ha="center", transform=ax1.transAxes, fontsize=9)
        
        ax2.imshow(preds[t])
        ax2.axis("off")
        ax2.text(0.5, -0.05, "Prediction", ha="center", transform=ax2.transAxes, fontsize=9)
        
        ax3.imshow(masks[t])
        ax3.axis("off")
        ax3.text(0.5, -0.05, "Ground truth", ha="center", transform=ax3.transAxes, fontsize=9)

        # Add caption above with prompt structure
        if caption:
            full_caption = f"Text Prompt: {caption}"
            fig.text(1, 0.95, full_caption, ha="center", fontsize=12)
        
        # Remove ALL margins and borders
        plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0, wspace=0, hspace=0)
        
        # Render the figure to an image
        fig.canvas.draw()
        
        # Convert the figure to a numpy array
        # Get the RGBA buffer from the figure canvas
        w, h = fig.canvas.get_width_height()
        
        # Use the correct method based on what's available
        try:
            # Try newer method first
            buf = fig.canvas.buffer_rgba()
            frame = np.asarray(buf).reshape(h, w, 4)
            frame = frame[:, :, :3]  # Drop alpha channel
        except AttributeError:
            try:
                # Try alternative method
                buf = fig.canvas.tostring_argb()
                frame = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
                # ARGB to RGB: skip alpha channel and reorder
                frame = frame[:, :, 1:4]
            except AttributeError:
                # Fallback
                buf = fig.canvas.renderer.buffer_rgba()
                frame = np.asarray(buf)[:, :, :3]
        
        frames.append(frame)
        plt.close(fig)

    imageio.mimsave(save_path, frames, fps=fps, loop=0)
    print(f"GIF saved to {save_path}")


# ──────────────────────  Attention module  ──────────────────────────────── #
class VisualGuidedCrossAttention(nn.Module):
    """Single‑head cross‑attention: visual **queries** × textual **keys/values**."""

    def __init__(self, feat_dim: int, text_dim: int, proj_dim: int) -> None:
        super().__init__()
        self.query_proj = nn.Conv2d(feat_dim, proj_dim, kernel_size=1)
        self.key_proj: nn.Linear = nn.Linear(text_dim, proj_dim)
        self.value_proj: nn.Linear = nn.Linear(text_dim, proj_dim)
        self.out_proj = nn.Conv2d(proj_dim, feat_dim, kernel_size=1)
        self.scale = float(proj_dim**0.5)

    def forward(self, feat: torch.Tensor, text: torch.Tensor) -> torch.Tensor:  # noqa: D401
        b, _, h, w = feat.shape
        queries = self.query_proj(feat).flatten(2).transpose(1, 2)  # [B, HW, P]
        keys = self.key_proj(text)  # [B, L, P]
        vals = self.value_proj(text)  # [B, L, P]

        attn = torch.softmax((queries @ keys.transpose(1, 2)) / self.scale, dim=-1)  # [B, HW, L]
        attended = (attn @ vals).transpose(1, 2).reshape(b, -1, h, w)
        return self.out_proj(attended)


# ─────────────────────────  Early Stopping  ──────────────────────────────── #
class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience=3, min_delta=0.0, verbose=True):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            min_delta (float): Minimum change to qualify as an improvement.
            verbose (bool): If True, prints a message for each improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.best_epoch = 0
        
    def __call__(self, val_loss, model, epoch, save_path):
        """
        Args:
            val_loss (float): Validation loss from current epoch.
            model (dict): Model state to save if improved.
            epoch (int): Current epoch number.
            save_path (str): Path to save the best model.
        """
        score = -val_loss  # Higher score is better
        
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(val_loss, model, epoch, save_path)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(val_loss, model, epoch, save_path)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model_state, epoch, save_path):
        """Save model when validation loss decreases."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving best model...')
        # Save as best_model.pth
        torch.save(model_state, os.path.join(save_path, f"best_model.pth"))
        # Also save with epoch number for reference
        # torch.save(model_state, os.path.join(save_path, f"best_model_epoch_{epoch}.pth"))
        self.val_loss_min = val_loss


# ───────────────────────────  Main training  ────────────────────────────── #
def main(args) -> None:  # noqa: C901 – big but clear
    # Set up constants from arguments
    data_dir = args.data_dir
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    num_brain_parts = args.num_brain_parts
    seed = args.seed
    model_name = args.model_name
    seq_len = args.seq_len
    text_dim = 768  # BioBERT-base hidden size
    embed_dim = text_dim
    encoded_dim = 512  # last UNet encoder channel dim (ResNet34 = 512)
    proj_dim = args.proj_dim
    
    # Set up directories
    checkpoint_dir = f"checkpoints-{num_brain_parts}"
    vis_dir = f"visuals-{num_brain_parts}/val"
    log_path = Path(f"log-{num_brain_parts}.txt")
    
    # Log the configuration
    print(f"Configuration:")
    print(f"- Number of brain parts: {num_brain_parts}")
    print(f"- Data directory: {data_dir}")
    print(f"- Batch size: {batch_size}")
    print(f"- Learning rate: {args.lr}")
    print(f"- Weight decay: {args.weight_decay}")
    print(f"- Early stopping patience: {args.patience}")
    print(f"- Checkpoints saved to: {checkpoint_dir}")
    print(f"- Visualizations saved to: {vis_dir}")
    print(f"- Log saved to: {log_path}")
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set up logging
    log_file = get_log_file(log_path)
    
    # Set random seed
    set_seed(seed)
    
    # Ensure directories exist
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    # ‑‑‑ data ‑‑‑
    train_ds = OASISDataset(data_dir, "train", num_brain_parts)
    val_ds = OASISDataset(data_dir, "val", num_brain_parts)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # ‑‑‑ text encoder (frozen) ‑‑‑
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModel.from_pretrained(model_name).to(device).eval()
    for p in bert_model.parameters():  # freeze
        p.requires_grad = False

    @torch.no_grad()
    def embed_text_batch(texts):
        tok = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=seq_len,
            return_tensors="pt",
        )
        out = bert_model(
            input_ids=tok["input_ids"].to(device),
            attention_mask=tok["attention_mask"].to(device),
        )
        return out.last_hidden_state.detach()  # [B, L, 768]

    # ‑‑‑ model & optimiser ‑‑‑
    model = Unet("resnet34", encoder_weights="imagenet", in_channels=1, classes=1).to(device)
    for p in model.encoder.parameters():
        p.requires_grad = False

    cross_attn = VisualGuidedCrossAttention(encoded_dim, text_dim, proj_dim).to(device)

    optimiser = optim.Adam(
        [
            {"params": list(model.decoder.parameters()) + list(model.segmentation_head.parameters()), "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": cross_attn.parameters(), "lr": args.lr, "weight_decay": args.weight_decay},
        ]
    )
    criterion = nn.BCEWithLogitsLoss()

    # ‑‑‑ checkpoint helpers ‑‑‑
    def latest_checkpoint() -> Tuple[str | None, int]:
        files = glob(os.path.join(checkpoint_dir, "checkpoint_*.pth"))
        if not files:
            return None, 0
        epochs = [int(re.search(r"checkpoint_(\d+)\.pth", f).group(1)) for f in files]
        best_ep = max(epochs)
        return os.path.join(checkpoint_dir, f"checkpoint_{best_ep}.pth"), best_ep

    def save_ckpt(epoch: int) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "cross_attn": cross_attn.state_dict(),
                "optim": optimiser.state_dict(),
            },
            os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pth"),
        )

    def load_ckpt(path: str) -> int:
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model"])
        cross_attn.load_state_dict(ckpt["cross_attn"])
        optimiser.load_state_dict(ckpt["optim"])
        return ckpt["epoch"]

    # ‑‑‑ maybe resume ‑‑‑
    ckpt_path, start_epoch = latest_checkpoint()
    if ckpt_path is not None:
        print(f"▶ Resuming from {ckpt_path}")
        start_epoch = load_ckpt(ckpt_path)

    # ════════════════════════  EPOCH LOOP  ════════════════════════════════ #
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(start_epoch + 1, num_epochs + 1):
        # ‑‑‑‑‑‑ train ‑‑‑‑‑‑
        model.train()
        cross_attn.train()
        tr_bce = tr_dice_loss = tr_dice = 0.0

        for imgs, masks, ref_txts in tqdm(train_loader, leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            imgs, masks = reshape_tensor(imgs), reshape_tensor(masks)
            txt_emb = embed_text_batch(ref_txts)

            optimiser.zero_grad()
            feats = model.encoder(imgs)
            feats[-1] = cross_attn(feats[-1], txt_emb.expand(feats[-1].shape[0], -1, -1))
            logits = model.segmentation_head(model.decoder(feats))

            bce = criterion(logits, masks)
            dice_c, dice_l = dice_coeff_and_loss(logits, masks)
            loss = bce + dice_l
            # loss = bce
            loss.backward()
            optimiser.step()

            tr_bce += bce.item()
            tr_dice_loss += dice_l.item()
            tr_dice += dice_c.item()

        # ‑‑‑‑‑‑ validation ‑‑‑‑‑‑
        model.eval()
        cross_attn.eval()
        val_bce = val_dice_loss = val_dice = 0.0
        with torch.no_grad():
            for v_idx, (imgs, masks, ref_txts) in enumerate(tqdm(val_loader, leave=False)):
                imgs, masks = imgs.to(device), masks.to(device)
                imgs, masks = reshape_tensor(imgs), reshape_tensor(masks)
                txt_emb = embed_text_batch(ref_txts)

                feats = model.encoder(imgs)
                feats[-1] = cross_attn(feats[-1], txt_emb.expand(feats[-1].shape[0], -1, -1))
                logits = model.segmentation_head(model.decoder(feats))
                
                bce = criterion(logits, masks)
                dice_c, dice_l = dice_coeff_and_loss(logits, masks)
                
                val_bce += bce.item()
                val_dice_loss += dice_l.item()
                val_dice += dice_c.item()

                # occasional GIFs
                if v_idx in {0, 1}:
                    save_prediction_vs_mask_gif(
                        imgs.cpu(),
                        logits.cpu(),
                        masks.cpu(),
                        save_path=os.path.join(vis_dir, f"epoch{epoch}_sample{v_idx}.gif"),
                        caption=ref_txts[0],  # ← show the reference text
                    )

        n_tr, n_val = len(train_loader), len(val_loader)
        val_total_loss = (val_bce / n_val) + (val_dice_loss / n_val)
        # val_total_loss = (val_bce / n_val)
        
        # Save current model state
        model_state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "cross_attn": cross_attn.state_dict(),
            "optim": optimiser.state_dict(),
            "val_loss": val_total_loss,
        }
        
        # Check if this is the best model so far
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            best_model_state = model_state
            # Save best model so far
            torch.save(model_state, os.path.join(checkpoint_dir, "best_model.pth"))
            log(f"New best model saved with validation loss: {best_val_loss:.6f}", log_file)
        
        log(
            f"Epoch {epoch:3d}/{num_epochs} | "
            f"train BCE {tr_bce / n_tr:.4f}  Dice {tr_dice / n_tr:.4f} | "
            f"val BCE {val_bce / n_val:.4f}  Dice {val_dice / n_val:.4f} | "
            f"val total loss {val_total_loss:.4f} | "
            f"best: {best_val_loss:.4f}",
            log_file
        )

        # Save regular checkpoint
        save_ckpt(epoch)
        
        # Early stopping check
        early_stopping(val_total_loss, model_state, epoch, checkpoint_dir)
        
        if early_stopping.early_stop:
            log(f"Early stopping triggered after epoch {epoch}", log_file)
            log(f"Best model was at epoch {early_stopping.best_epoch} with loss {early_stopping.val_loss_min:.6f}", log_file)
            break

    # At the end of training, ensure we have the best model saved
    if best_model_state is not None:
        best_epoch = best_model_state["epoch"]
        log(f"Training complete. Best model was at epoch {best_epoch} with loss {best_val_loss:.6f}", log_file)
        # Save a copy with the epoch number for clarity
        torch.save(best_model_state, os.path.join(checkpoint_dir, f"best_model_epoch_{best_epoch}.pth"))
    
    log("✔ Training complete.", log_file)
    log_file.close()


# ─────────────────────────────  Entry point  ────────────────────────────── #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train UNet‑ResNet34 with visual‑guided cross‑attention"
    )
    parser.add_argument(
        "--num-brain-parts",
        type=int,
        default=2,
        help="How many anatomical parts to segment (affects checkpoint/log paths)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="oasis-redefined",
        help="Directory containing the dataset",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="Number of epochs to train",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="dmis-lab/biobert-base-cased-v1.1",
        help="Pretrained model name for text encoder",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=10,
        help="Maximum sequence length for text input",
    )
    parser.add_argument(
        "--proj-dim",
        type=int,
        default=128,
        help="Projection dimension for cross-attention",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay for optimizer",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Patience for early stopping",
    )
    
    args = parser.parse_args()
    main(args)

# python finetune_with_bert_updated.py --num-brain-parts 4