# Social Guide Model Code

This folder contains scripts to generate synthetic examples, fine-tune a LLaMA-style model with LoRA, and run inference/evaluation for social media rule violation detection.

## Folder map
- gen_aii_train.py: Validates JSONL format, builds SFT training text, and fine-tunes a base model with LoRA.
- run_gen_ai.py: Runs batch inference from a text file and saves responses to a report file.
- run_eval.py: Samples random JSONL entries and prints model responses with ground truth.
- enhanced_gen_ai_train_eval.py: Inference plus optional evaluation with metrics and reports.
- updated_run_gen_ai_train_eval.py: Older eval script with random JSONL sampling and metrics (update file paths before use).
- new_llama_generator.py: Generates synthetic JSON examples per rule using a base Llama model.

Data and outputs in this folder:
- support_converted.jsonl: Main training/eval dataset with input/output/id fields.
- data.jsonl: Example JSONL entries in the same schema (no id field).
- input.txt: One-sentence-per-line input for inference.
- model_responses.txt: Example inference output file.
- analysis_results.json, processing.log: Example results/logs produced by scripts.

## Data format
Most scripts expect each JSONL line to follow this schema:

```
{"input": "text string", "output": {"violation": true/false, "rule": "rule text", "explanation": "why"}, "id": 0}
```

Notes:
- gen_aii_train.py also accepts input objects where `input` is a dict containing a `post` field.
- enhanced_gen_ai_train_eval.py can read labeled plain-text lines as `sentence|true` or `sentence,false` when `--evaluate` is used.

## Setup
- Python 3.9+ recommended.
- Install dependencies:

```
pip install torch transformers datasets peft trl scikit-learn tqdm numpy
```

- Some scripts assume a GPU and use `cuda:0` directly. Adjust device settings if running on CPU.
- Replace the placeholder `token="your-key"` in `gen_aii_train.py` and `new_llama_generator.py` with a valid Hugging Face token if required by your model.

## Workflows

### 1) Fine-tune a model
Uses `support_converted.jsonl` by default and trains a LoRA adapter:

```
python gen_aii_train.py support_converted.jsonl
```

Outputs a model directory (default: `./llama-2-7b-chat-violation-checker-light`).

### 2) Run inference on plain sentences
Reads `input.txt`, formats prompts, and saves responses to `model_responses.txt`:

```
python run_gen_ai.py --model ./llama-2-7b-chat-violation-checker-light --input input.txt --output model_responses.txt
```

### 3) Evaluate with labeled input
Provide a labeled file where each line is `sentence|true` or `sentence|false`:

```
python enhanced_gen_ai_train_eval.py --model ./llama-2-7b-chat-violation-checker --input labeled.txt --evaluate
```

This prints predictions and writes metrics plus per-sample outputs to the `--output` file (default: `model_responses.txt`).

### 4) Quick random-sample eval from JSONL
Samples entries from a JSONL dataset and compares predictions to ground truth:

```
python run_eval.py
python updated_run_gen_ai_train_eval.py
```

Update the JSONL path in `updated_run_gen_ai_train_eval.py` before running.

### 5) Generate synthetic examples from rules
Creates new JSON examples per rule using a base Llama model:

```
python new_llama_generator.py --input_file support_converted.jsonl --output_file synthetic.json --model_name meta-llama/Llama-3.1-8B
```

Generated examples are saved as a JSON array in `synthetic.json` and logs are written to `processing.log`.

## Notes and troubleshooting
- If you see tokenizer padding warnings, verify `tokenizer.pad_token` is set (handled in `gen_aii_train.py`).
- If outputs do not include an explicit `violation` statement, `enhanced_gen_ai_train_eval.py` falls back to keyword heuristics.
- Check model paths in each script; they are hard-coded defaults and may need to be updated for your environment.
