# Team 9 - Private Data Generation (DP-PrefSyn)

This folder contains scripts that generate synthetic preference data under differential privacy (DP). The pipeline embeds preference pairs, applies DP-PCA + DP-SGD to learn reward directions, then uses a local LLM (via vLLM) to synthesize responses. A separate script estimates total privacy loss.

## Folder map
- DPPrefSyn_HH.py: DP preference synthesis for the Anthropic HH-RLHF dataset.
- DPPrefSyn_OA.py: DP preference synthesis for OpenAssistant (OASST1).
- DPPrefSyn_summarize.py: DP preference synthesis for summarize-from-feedback.
- privacy_analysis.py: Computes composed epsilon using PRV accountant.
- t9-gaofengyu_1295565_15144352_report.pdf: Report.
- t9-shenwei_6995_15146822_GenAI Project.pptx: Slides.

## What the scripts do
1. Load a preference dataset from Hugging Face.
2. Embed chosen/rejected pairs with a sentence transformer.
3. Compute embedding differences and apply DP-PCA.
4. Cluster embeddings and learn reward directions with (DP-)SGD.
5. Sample public prompts and generate synthetic responses via vLLM.
6. (Optional) Estimate total privacy loss in `privacy_analysis.py`.

## Important library: vLLM
The generators use `vllm.LLM` to run a local Llama model for fast batch decoding.

### Install vLLM (Linux recommended)

```
pip install vllm
```

Notes:
- vLLM expects a CUDA-capable GPU.
- For non-GPU environments, you will need to replace vLLM with a different inference backend.

## Setup
Minimum dependencies (all scripts):

```
pip install torch datasets sentence-transformers scikit-learn matplotlib tqdm numpy
```

DP + privacy accounting dependencies:

```
pip install opacus opendp prv-accountant
```

vLLM dependency:

```
pip install vllm
```

## Running the scripts
Each script accepts `--eps` (privacy budget) and `--run` (seed/run id).

Examples:

```
python DPPrefSyn_HH.py --eps 8 --run 1
python DPPrefSyn_OA.py --eps 4 --run 1
python DPPrefSyn_summarize.py --eps 8 --run 1
```

Privacy accounting:

```
python privacy_analysis.py
```

## Outputs
Each generator writes a results folder:
- `results_HH/`
- `results_OA/`
- `results_summary/`

The synthetic responses are collected in-memory in each script; add a write step if you want to persist them.

## Notes and gotchas
- These scripts download datasets and models from Hugging Face on first run.
- vLLM runs the model `meta-llama/Llama-2-7b-chat-hf` by default; update the model string if you use a different checkpoint.
- DP settings are hard-coded for `eps` 4 or 8. Other values will need new hyperparameters.
- `opendp` enables experimental features; installation can be sensitive to platform/Python version.

