# Team 12 - LLaMA Guard Fine-Tuning for Toxicity

This folder contains code and assets for fine-tuning a LLaMA Guard model for safety classification (safe/unsafe + category). It includes training scripts, data conversion utilities, a prediction helper, and a simple Flask UI.

## Folder map
- team12-code/fine-tune/: Main code and data.
- t12-toxicity.pptx: Slides.
- t12-toxicity_AI_risks.pdf: Report.

Inside `team12-code/fine-tune/`:
- scripts/finetune.py: LoRA fine-tuning script with Hugging Face Trainer.
- src/predict.py: Prediction helper with custom prompt formatting.
- src/prompt_builder.py: Builds the safety policy prompt and conversation format.
- configs/finetune_config.py: Training hyperparameters and paths.
- configs/safety_categories.py: Safety category definitions.
- data/: Training and test JSON data.
- data/convert.py: Converts raw data to training format.
- gui.py: Flask demo UI for inference.
- predict_report.py: Script for generating evaluation reports.
- accelerate_config.yaml: Accelerate configuration.

## How it works
1. Build safety prompts using a fixed policy and category list.
2. Fine-tune `meta-llama/Llama-Guard-3-1B` with LoRA.
3. Run inference with a custom prompt to output `safe`/`unsafe` and a category code.
4. Optionally run a Flask web UI for manual testing.

## Setup
Create a Python environment with the required libraries.

```
pip install torch transformers datasets peft accelerate
pip install flask
```

## Important library: Hugging Face Transformers
All training and inference are built on Transformers.

Install:

```
pip install transformers
```

If you use LoRA, also install PEFT:

```
pip install peft
```

## Training
From the `team12-code/fine-tune/` directory:

```
python scripts/finetune.py
```

Adjust paths and hyperparameters in `configs/finetune_config.py`.

## Inference (CLI)
Use the predictor in `src/predict.py` via your own script, or run the Flask UI.

## Flask UI
Update the model path in `gui.py` before running:

```
python gui.py
```

Then visit `http://localhost:5000`.

## Notes and gotchas
- The code expects a GPU for training and inference.
- The tokenizer path in `gui.py` is hard-coded; update it to your local fine-tuned model.
- The data format is JSON with `conversation` objects; see files in `data/`.

