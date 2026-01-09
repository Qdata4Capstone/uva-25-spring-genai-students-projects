# Team 6 - LLM Code Benchmarking

This folder benchmarks multiple LLMs on the HumanEval+ code generation dataset. The notebooks load problems, request code from each model, run unit tests, and summarize pass@1 results with plots and CSVs.

## Folder contents
- team6-GenAI_Project-codeGen.ipynb: Main benchmarking notebook (code generation focus).
- team6-GenAI_Project.ipynb: Same benchmarking flow with identical core logic.

## What the notebooks do
1. Install dependencies (OpenAI client, Anthropic, tqdm).
2. Load `HumanEvalPlus.jsonl` from the local folder.
3. Query multiple models (OpenAI, Anthropic, xAI) using OpenAI-compatible clients.
4. Extract Python code from model responses.
5. Execute unit tests with a timeout and record pass/fail.
6. Save per-model results JSON, a CSV summary, and a comparison plot.

## Important library: OpenAI Python SDK
The notebooks use the OpenAI SDK to talk to OpenAI, Anthropic, and xAI via compatible endpoints.

### Install

```
pip install openai anthropic
```

### API keys
You need API keys for any models you want to run:
- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- xAI: `XAI_API_KEY`

In Colab, the notebooks expect keys via `google.colab.userdata`:

```
from google.colab import userdata
userdata.get('OPENAI_API_KEY')
```

For local runs, set environment variables before starting Jupyter:

```
export OPENAI_API_KEY=... 
export ANTHROPIC_API_KEY=...
export XAI_API_KEY=...
```

## Dataset requirement
Place `HumanEvalPlus.jsonl` in this folder before running. The notebook raises `FileNotFoundError` if it is missing.

## Setup
Minimum dependencies:

```
pip install openai anthropic tqdm matplotlib pandas
```

## Run
Open either notebook and run all cells in order:

- `team6-GenAI_Project-codeGen.ipynb`
- `team6-GenAI_Project.ipynb`

The default models are:
- `o3` (OpenAI)
- `claude` (Anthropic)
- `grok` (xAI)

Edit the `model_map` in `generate_code()` to change model names.

## Outputs
Running the full benchmark writes:
- `*_humaneval_results.json` (per-model detailed results)
- `benchmark_results.csv` (summary)
- `model_comparison.png` (bar chart)

## Notes and gotchas
- The execution step uses `signal.alarm`, which may not behave the same on Windows.
- Rate limits are handled with a simple retry/backoff based on error messages.
- Long runs can be slow; consider reducing the dataset size or adding delays per request.

