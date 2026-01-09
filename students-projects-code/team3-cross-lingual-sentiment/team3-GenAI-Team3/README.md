# Cross-Lingual Sentiment Alignment (Team 3)

This folder contains code to generate multilingual news responses, back-translate to English, and evaluate how well sentiment aligns across languages. It also includes a fine-tuning pipeline for a Qwen-style model and a small evaluation harness built on multilingual RoBERTa sentiment scoring.

## Folder map
- main.py: Fine-tuning workflow using Accelerate + Hugging Face Trainer. Includes a demo block that calls the generation/eval pipeline.
- parallel_inference.py: End-to-end pipeline (translate -> generate -> back-translate -> sentiment), plus a parallel inference runner that saves per-topic results.
- run_parallel_inference.sh: Multi-GPU launcher for parallel inference (hard-coded conda path).
- run_eval.py: Evaluates generated samples vs. reference bundles and writes plots + per-topic metrics.
- sentiment_eval.py: Sentiment scoring and evaluation utilities based on `twitter-xlm-roberta-base-sentiment`.
- gen_dicts.py: Sample generated paragraphs per topic/language.
- ground_truth_en.py, ground_truth_fr.py, ground_truth_ja.py, ground_truth_zh.py: Reference bundles per topic for evaluation.
- english.txt, french.py, japanese.py, chinese.py: Alternate reference bundles (note: english.txt is a Python snippet saved as .txt).
- news_topics.json, news_topics_small.json: Topics and example prompts for inference.
- datafetch/: News crawler to build multilingual training data.
- main.ipynb: Notebook version of the pipeline and plotting.
- processed_sentiment_data.csv, attitude_by_language.csv, positive_sentiment_bar_chart.png: Example outputs from analysis.

## Setup
Install core dependencies:

```
pip install -r requirements.txt
```

Additional dependencies (used by the crawler/pipeline):

```
pip install accelerate datasets scipy requests beautifulsoup4 lxml google-cloud-translate tqdm
```

Notes:
- First run will download models from Hugging Face.
- GPU is strongly recommended for fine-tuning and generation.
- `parallel_inference.py` requires a Google Translate API key (see Configuration below).

## Configuration
- `parallel_inference.py` uses `API_KEY = "your-key"` for Google Translate. Replace with a valid key.
- Model names and training hyperparameters live in the `config` dict inside `main.py` and `parallel_inference.py`.
- `ds_config.json` exists for DeepSpeed-style settings but is not wired into the scripts.

## Workflows

### 1) Run sentiment evaluation on provided samples
This compares generated samples (from `gen_dicts.py`) to reference bundles and writes plots and metrics to `results/`.

```
python run_eval.py
```

Outputs:
- `results/conf_*.png` (confusion matrices)
- `results/accuracy_bar.png`
- `results/per_topic_metrics.csv`

### 2) Run the translation/generation pipeline for a topic
This runs the pipeline for one topic/example index and saves a pickle with all languages.

```
python parallel_inference.py --gpu_num 0 --topic_idx 0 --example_idx 0
```

Outputs:
- `result/<topic>-<example>.pkl`

### 3) Fine-tune the generator
Fine-tunes on JSONL datasets under `./dataset` (each file is a language). Edit the config at the bottom of `main.py`.

```
python main.py
```

### 4) Crawl news data (optional)
The crawler writes JSONL files to `datafetch/output/`.

```
python datafetch/main.py --lang en
python datafetch/main.py --lang zh
```

## Data expectations
- Training data: JSONL with fields `title`, `content`, `publish_time`, and `url` per row.
- Evaluation data: DataFrame with `topic`, `generated` (blank), and `references` (list of strings).

## Gotchas
- `run_eval.py` imports `english`, `french`, `japanese`, and `chinese` modules. In this folder, `english.txt` is a Python snippet saved as a .txt file, and `ground_truth_en.py` is the canonical English file. Either rename `english.txt` to `english.py` or change the import in `run_eval.py` to use `ground_truth_en.py`.
- `main.py` calls `pipeline_fn` but does not import it. Use `from parallel_inference import pipeline_fn` or run the notebook if you want the full pipeline demo.
- `run_parallel_inference.sh` uses a hard-coded conda path and GPU count; update it for your environment.

## Key files to start with
- For evaluation and plots: `run_eval.py`
- For the full multilingual pipeline: `parallel_inference.py`
- For training: `main.py`
- For data collection: `datafetch/main.py`
