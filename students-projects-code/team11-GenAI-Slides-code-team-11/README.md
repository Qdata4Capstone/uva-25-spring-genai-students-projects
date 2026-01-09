# Team 11 - TESSERACT Brain Segmentation

This folder contains code for a text-guided brain MRI segmentation project using a UNet backbone and BERT-based text embeddings. It includes training, evaluation, and data preparation utilities, plus a demo notebook.

## Folder map
- TESSERACT-code-Team-11/: Source code and notebook.
- Team-11-presentation.pptx: Slides.
- t11-paulrishov_1284582_15146690_TESSERACT-Team11-project-report.pdf: Report.

Inside `TESSERACT-code-Team-11/`:
- finetune_with_bert_updated.py: Training loop for text-guided UNet segmentation.
- test_with_bert_updated.py: Evaluation/inference with GIF visualizations.
- data_loader.py: OASIS dataset loader and preprocessing.
- separate_brain_part_files.py: Utility to filter and copy data by brain part.
- demo_presentation-2.ipynb: Demo notebook used for presentation.

## How it works
1. Loads OASIS MRI volumes from `.npy` files.
2. Encodes text prompts with a Transformer model (BERT) to guide segmentation.
3. Trains a UNet to predict the mask for the requested brain region.
4. Evaluates and saves GIFs comparing prediction vs. ground truth.

## Setup
Create a Python environment with PyTorch and the required libraries.

```
pip install torch torchvision torchaudio
pip install transformers segmentation-models-pytorch imageio matplotlib numpy tqdm
```

## Important library: segmentation-models-pytorch
The UNet backbone is provided by `segmentation-models-pytorch`.

Install:

```
pip install segmentation-models-pytorch
```

If you run into CUDA issues, make sure your PyTorch build matches your CUDA version.

## Data expectations
The scripts expect an OASIS-style dataset directory with `train/`, `val/`, and `test/` splits. Each file is a `.npy` dict with keys:
- `voxal_video`
- `filtered_mask`
- `text_prompt`

Example structure:

```
oasis-dataset/
  train/
  val/
  test/
```

Use `separate_brain_part_files.py` to filter by brain region and copy to a new split directory.

## Training
Run training with the default settings in the script:

```
python TESSERACT-code-Team-11/finetune_with_bert_updated.py
```

Adjust paths and hyperparameters inside the script (dataset folder, batch size, epochs).

## Evaluation
Run inference and visualization:

```
python TESSERACT-code-Team-11/test_with_bert_updated.py
```

This script saves GIFs that compare image, prediction, and ground truth.

## Notes and gotchas
- Large 3D volumes require GPU memory; consider smaller batches if you see OOM.
- The code slices the volume for training; review `data_loader.py` if you want different slicing.
- Paths are hard-coded in places; update them to match your local dataset.

