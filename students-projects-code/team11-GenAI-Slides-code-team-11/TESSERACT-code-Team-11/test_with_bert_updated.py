import os
import argparse
import re
from glob import glob
from typing import Tuple

import imageio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from segmentation_models_pytorch import Unet
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Import your dataset class - make sure this file is in the same directory
from data_loader import OASISDataset


# ───────────────────────────────  Utilities  ─────────────────────────────── #
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


def save_prediction_vs_mask_gif_bef(
    images: torch.Tensor,
    logits: torch.Tensor,
    masks: torch.Tensor,
    *,
    save_path: str,
    caption: str = "",
    fps: int = 5,
) -> None:
    """Save a GIF with *Image | Prediction | Ground‑truth* panels."""
    images = np.rot90(images.squeeze(1).cpu().numpy(), k=1, axes=(1, 2))
    preds = (torch.sigmoid(logits) > 0.5).float().squeeze(1).cpu().numpy()
    preds = np.rot90(preds, k=1, axes=(1, 2))
    masks = np.rot90(masks.squeeze(1).cpu().numpy(), k=1, axes=(1, 2))

    frames = []
    for t in range(images.shape[0]):
        # Create a figure with a slightly larger height to accommodate labels
        fig = plt.figure(figsize=(7.5, 3.0))

        # Create grid with no spacing between panels
        gs = fig.add_gridspec(nrows=1, ncols=3, wspace=0.01, hspace=0)

        # Add subplots
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])

        # Plot images with no padding
        # Image
        ax1.imshow(images[t], cmap="gray")
        ax1.set_xticks([]); ax1.set_yticks([])
        ax1.set_xlabel("Image", fontsize=9, labelpad=4)

        # Prediction
        ax2.imshow(preds[t])
        ax2.set_xticks([]); ax2.set_yticks([])
        ax2.set_xlabel("Prediction", fontsize=9, labelpad=4)

        # Ground truth
        ax3.imshow(masks[t])
        ax3.set_xticks([]); ax3.set_yticks([])
        ax3.set_xlabel("Ground truth", fontsize=9, labelpad=4)

        # Add caption centered at the top with minimal space
        if caption:
            # Add title to the figure for the caption - centered above all panels
            fig.suptitle(f"Text Prompt: {caption}", fontsize=11, y=0.98)

        # Adjust layout manually with increased bottom padding
        plt.subplots_adjust(left=0, right=1, top=0.85, bottom=0.2, wspace=0.01, hspace=0)

        # Render the figure to an image
        fig.canvas.draw()

        # Convert the figure to a numpy array
        try:
            w, h = fig.canvas.get_width_height()
            buf = fig.canvas.buffer_rgba()
            frame = np.asarray(buf).reshape(h, w, 4)
            frame = frame[:, :, :3]  # Drop alpha channel
        except AttributeError:
            try:
                w, h = fig.canvas.get_width_height()
                buf = fig.canvas.tostring_argb()
                frame = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
                frame = frame[:, :, 1:4]  # ARGB to RGB
            except AttributeError:
                buf = fig.canvas.renderer.buffer_rgba()
                frame = np.asarray(buf)[:, :, :3]

        frames.append(frame)
        plt.close(fig)

    imageio.mimsave(save_path, frames, fps=fps, loop=0)
    print(f"GIF saved to {save_path}")


def save_prediction_vs_mask_gif(
    images: torch.Tensor,
    logits: torch.Tensor,
    masks: torch.Tensor,
    *,
    save_path: str,
    caption: str = "",
    fps: int = 5,
) -> None:
    """
    Save a GIF with 4 panels: Image | Prediction | Ground Truth | Overlay.
    Segmented areas are colored based on caption; background is black.
    """

    # Part-specific color
    color_map = {
        "Left-Cerebral-White-Matter": [255, 0, 0],
        "Right-Cerebral-White-Matter": [0, 0, 255],
        "Left-Thalamus": [0, 255, 0],
        "Right-Thalamus": [255, 255, 0],
    }
    color = [255, 255, 255]
    for key in color_map:
        if key in caption:
            color = color_map[key]
            break
    color = np.array(color) / 255.0

    # Prepare data: shape [T, H, W]
    images = np.rot90(images.squeeze(1).cpu().numpy(), k=1, axes=(1, 2))
    preds = (torch.sigmoid(logits) > 0.5).float().squeeze(1).cpu().numpy()
    preds = np.rot90(preds, k=1, axes=(1, 2))
    masks = np.rot90(masks.squeeze(1).cpu().numpy(), k=1, axes=(1, 2))

    frames = []
    for t in range(images.shape[0]):
        fig = plt.figure(figsize=(12, 3.0))
        gs = fig.add_gridspec(nrows=1, ncols=4, wspace=0.01, hspace=0)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[0, 3])

        # Image panel
        ax1.imshow(images[t], cmap="gray")
        ax1.set_xticks([]); ax1.set_yticks([])
        ax1.set_xlabel("Image", fontsize=9, labelpad=4)

        # Prediction panel (black background with color mask)
        pred_rgb = np.zeros((*preds[t].shape, 3))
        for c in range(3):
            pred_rgb[:, :, c] = preds[t] * color[c]
        ax2.imshow(pred_rgb)
        ax2.set_xticks([]); ax2.set_yticks([])
        ax2.set_xlabel("Prediction", fontsize=9, labelpad=4)

        # Ground truth panel (black background with color mask)
        mask_rgb = np.zeros((*masks[t].shape, 3))
        for c in range(3):
            mask_rgb[:, :, c] = masks[t] * color[c]
        ax3.imshow(mask_rgb)
        ax3.set_xticks([]); ax3.set_yticks([])
        ax3.set_xlabel("Ground truth", fontsize=9, labelpad=4)

        # Overlay panel
        ax4.imshow(images[t], cmap="gray")
        ax4.imshow(pred_rgb, alpha=0.4)
        ax4.set_xticks([]); ax4.set_yticks([])
        ax4.set_xlabel("Overlay", fontsize=9, labelpad=4)

        if caption:
            fig.suptitle(f"Text Prompt: {caption}", fontsize=11, y=0.98)

        plt.subplots_adjust(left=0, right=1, top=0.85, bottom=0.2, wspace=0.01, hspace=0)

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = fig.canvas.buffer_rgba()
        frame = np.asarray(buf).reshape(h, w, 4)
        frame = frame[:, :, :3]

        frames.append(frame)
        plt.close(fig)

    imageio.mimsave(save_path, frames, fps=fps, loop=0)
    print(f"GIF saved to {save_path}")


def save_static_comparison_plot(
    images: torch.Tensor,
    logits: torch.Tensor,
    masks: torch.Tensor,
    *,
    save_path: str,
    caption: str = "",
    frame_stride: int = 4,
) -> None:
    # Convert and rotate 90° clockwise
    images = np.rot90(images.squeeze(1).cpu().numpy(), k=1, axes=(1, 2))
    preds = (torch.sigmoid(logits) > 0.5).float().squeeze(1).cpu().numpy()
    preds = np.rot90(preds, k=1, axes=(1, 2))
    masks = np.rot90(masks.squeeze(1).cpu().numpy(), k=1, axes=(1, 2))

    total_frames = images.shape[0]
    selected_indices = list(range(0, total_frames, frame_stride))
    n_cols = len(selected_indices)

    # Part-specific color
    color_map = {
        "Left-Cerebral-White-Matter": [255, 0, 0],
        "Right-Cerebral-White-Matter": [0, 0, 255],
        "Left-Thalamus": [0, 255, 0],
        "Right-Thalamus": [255, 255, 0],
    }
    seg_color = [255, 255, 255]
    for key in color_map:
        if key in caption:
            seg_color = color_map[key]
            break

    def overlay_color(mask: np.ndarray, color: list) -> np.ndarray:
        h, w = mask.shape
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(3):
            img[:, :, i] = mask * color[i]
        return img

    row_titles = ["Image", "Prediction", "Ground truth"]

    # Create figure and GridSpec
    fig = plt.figure(figsize=(n_cols * 1.8, 6))
    gs = gridspec.GridSpec(3, n_cols, wspace=0.0, hspace=0.05)

    for row_idx in range(3):
        for col_idx, frame_idx in enumerate(selected_indices):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.axis("off")

            if row_idx == 0:
                ax.imshow(images[frame_idx], cmap="gray")
                ax.set_title(f"Frame {frame_idx}", fontsize=9, pad=4)
            elif row_idx == 1:
                ax.imshow(overlay_color(preds[frame_idx], seg_color))
            else:
                ax.imshow(overlay_color(masks[frame_idx], seg_color))

            # Row labels on the far-left column only
            if col_idx == 0:
                ax.text(
                    -0.25, 0.5, row_titles[row_idx],
                    va="center", ha="right", rotation=90,
                    fontsize=11, transform=ax.transAxes
                )

    # Caption on top
    fig.suptitle(f"Text Prompt: {caption}", fontsize=13, weight="bold", y=0.99)

    # Adjust spacing
    plt.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.03)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Static plot saved to {save_path}")



# ──────────────────────────  Attention module  ──────────────────────────────── #
class VisualGuidedCrossAttention(nn.Module):
    """Single‑head cross‑attention: visual **queries** × textual **keys/values**."""

    def __init__(self, feat_dim: int, text_dim: int, proj_dim: int) -> None:
        super().__init__()
        self.query_proj = nn.Conv2d(feat_dim, proj_dim, kernel_size=1)
        self.key_proj: nn.Linear = nn.Linear(text_dim, proj_dim)
        self.value_proj: nn.Linear = nn.Linear(text_dim, proj_dim)
        self.out_proj = nn.Conv2d(proj_dim, feat_dim, kernel_size=1)
        self.scale = float(proj_dim**0.5)

    def forward(self, feat: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        b, _, h, w = feat.shape
        queries = self.query_proj(feat).flatten(2).transpose(1, 2)  # [B, HW, P]
        keys = self.key_proj(text)  # [B, L, P]
        vals = self.value_proj(text)  # [B, L, P]

        attn = torch.softmax((queries @ keys.transpose(1, 2)) / self.scale, dim=-1)  # [B, HW, L]
        attended = (attn @ vals).transpose(1, 2).reshape(b, -1, h, w)
        return self.out_proj(attended)


def test_model(args):
    # Set up constants from arguments
    data_dir = args.data_dir
    batch_size = args.batch_size
    num_brain_parts = args.num_brain_parts
    model_name = args.model_name
    seq_len = args.seq_len
    text_dim = 768  # BioBERT-base hidden size
    encoded_dim = 512  # last UNet encoder channel dim (ResNet34 = 512)
    proj_dim = args.proj_dim
    
    # Set up directories
    checkpoint_dir = f"checkpoints-{num_brain_parts}"
    results_dir = f"test_results-{num_brain_parts}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ‑‑‑ data ‑‑‑
    test_ds = OASISDataset(data_dir, f"test-{num_brain_parts}", num_brain_parts)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    print(f"Test dataset size: {len(test_ds)}")

    # ‑‑‑ text encoder (frozen) ‑‑‑
    print(f"Loading text encoder: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModel.from_pretrained(model_name).to(device).eval()
    
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

    # ‑‑‑ model ‑‑‑
    model = Unet("resnet34", encoder_weights="imagenet", in_channels=1, classes=1).to(device)
    cross_attn = VisualGuidedCrossAttention(encoded_dim, text_dim, proj_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    
    # ‑‑‑ load best model ‑‑‑
    def find_best_model():
        # Check if best_model.pth exists
        best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
        if os.path.exists(best_model_path):
            return best_model_path
            
        # Check for best_model_epoch_X.pth
        files = glob(os.path.join(checkpoint_dir, "best_model_epoch_*.pth"))
        if files:
            return files[0]
            
        # Fall back to latest checkpoint
        files = glob(os.path.join(checkpoint_dir, "checkpoint_*.pth"))
        if not files:
            raise FileNotFoundError(f"No model checkpoints found in {checkpoint_dir}")
        
        epochs = [int(re.search(r"checkpoint_(\d+)\.pth", f).group(1)) for f in files]
        best_ep = max(epochs)
        return os.path.join(checkpoint_dir, f"checkpoint_{best_ep}.pth")
    
    best_model_path = find_best_model()
    print(f"Loading best model from: {best_model_path}")
    
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    cross_attn.load_state_dict(checkpoint["cross_attn"])
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # ‑‑‑ testing ‑‑‑
    model.eval()
    cross_attn.eval()
    test_bce = test_dice_loss = test_dice = 0.0
    
    print("Starting evaluation on test set...")
    textwise_stats = defaultdict(lambda: {"dice": 0.0, "dice_loss": 0.0, "bce": 0.0, "count": 0})
    with torch.no_grad():
        for t_idx, (imgs, masks, ref_txts) in enumerate(tqdm(test_loader)):
            imgs, masks = imgs.to(device), masks.to(device)
            imgs, masks = reshape_tensor(imgs), reshape_tensor(masks)
            txt_emb = embed_text_batch(ref_txts)

            feats = model.encoder(imgs)
            feats[-1] = cross_attn(feats[-1], txt_emb.expand(feats[-1].shape[0], -1, -1))
            logits = model.segmentation_head(model.decoder(feats))
            
            bce = criterion(logits, masks)
            dice_c, dice_l = dice_coeff_and_loss(logits, masks)
            
            ref_text = ref_txts[0]  # assuming batch_size = 1
            test_bce += bce.item()
            test_dice_loss += dice_l.item()
            test_dice += dice_c.item()

            # Accumulate per-text statistics
            textwise_stats[ref_text]["dice"] += dice_c.item()
            textwise_stats[ref_text]["dice_loss"] += dice_l.item()
            textwise_stats[ref_text]["bce"] += bce.item()
            textwise_stats[ref_text]["count"] += 1

            # Save visualization for each test sample or a subset
            if t_idx < args.num_visualizations:
                save_prediction_vs_mask_gif(
                    imgs.cpu(),
                    logits.cpu(),
                    masks.cpu(),
                    save_path=os.path.join(results_dir, f"test_sample_{t_idx}.gif"),
                    caption=ref_txts[0],  # Show the reference text
                )
                save_static_comparison_plot(
                    imgs.cpu(),
                    logits.cpu(),
                    masks.cpu(),
                    save_path=os.path.join(results_dir, f"comparison_plot_{t_idx}.png"),
                    caption=ref_txts[0],
                    frame_stride=4,
                )

    
    n_test = len(test_loader)
    avg_bce = test_bce / n_test
    avg_dice_coeff = test_dice / n_test
    avg_dice_loss = test_dice_loss / n_test
    total_loss = avg_bce + avg_dice_loss
    
    print("\nTest Results:")
    print(f"BCE Loss: {avg_bce:.4f}")
    print(f"Dice Coefficient: {avg_dice_coeff:.4f}")
    print(f"Dice Loss: {avg_dice_loss:.4f}")
    print(f"Total Loss: {total_loss:.4f}")
    
    # Save results to file
    with open(os.path.join(results_dir, "test_results.txt"), "w") as f:
        f.write(f"Test Results:\n")
        f.write(f"BCE Loss: {avg_bce:.4f}\n")
        f.write(f"Dice Coefficient: {avg_dice_coeff:.4f}\n")
        f.write(f"Dice Loss: {avg_dice_loss:.4f}\n")
        f.write(f"Total Loss: {total_loss:.4f}\n")
        f.write(f"Model: {best_model_path}\n")
        f.write(f"Number of test samples: {len(test_ds)}\n")
    
    print("\nPer-text Results:")
    with open(os.path.join(results_dir, "textwise_results.txt"), "w") as f:
        f.write("Per-text Results:\n")
        for txt, stats in textwise_stats.items():
            count = stats["count"]
            avg_dice = stats["dice"] / count
            avg_dice_loss = stats["dice_loss"] / count
            avg_bce = stats["bce"] / count
            total = avg_dice_loss + avg_bce

            line = (
                f"Text Prompt: {txt}\n"
                f"  Samples: {count}\n"
                f"  Dice Coeff: {avg_dice:.4f}, Dice Loss: {avg_dice_loss:.4f}, "
                f"BCE: {avg_bce:.4f}, Total Loss: {total:.4f}\n"
            )
            print(line)
            f.write(line + "\n")

    print(f"Results saved to {os.path.join(results_dir, 'test_results.txt')}")
    print("✔ Testing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test UNet‑ResNet34 with visual‑guided cross‑attention"
    )
    parser.add_argument(
        "--num-brain-parts",
        type=int,
        default=4,
        help="How many anatomical parts to segment (affects checkpoint/results paths)",
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
        help="Batch size for testing",
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
        "--num-visualizations",
        type=int,
        default=10,
        help="Number of test samples to visualize",
    )
    
    args = parser.parse_args()
    test_model(args)