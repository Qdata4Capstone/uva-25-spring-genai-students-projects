# Team 5 - Neuron Disambiguation (Explain LLM)

This folder contains a single notebook that studies neuron behavior in a tiny TransformerLens model. It visualizes neuron activations, inspects output-token biases, and tests how ablation of specific neurons changes the model’s predictions across different contexts.

## Folder contents
- team5_nguyenmatthew_1258025_15148875_Neuron_Disambiguation.ipynb: Main analysis notebook.

## What the notebook does
1. Loads a small TransformerLens model (`gelu-1l`) and hooks into MLP activations.
2. Visualizes the max-activation token for a selected neuron (default: layer 0, neuron 2029).
3. Shows tokens that the neuron most boosts or suppresses via the unembedding matrix.
4. Finds co-activating neurons for different prompts.
5. Performs neuron ablation and compares top-k logits before/after ablation.
6. Identifies “canceller” neurons that offset a target neuron in different contexts and tests their causal effect.

## Important library: TransformerLens
TransformerLens is required for the HookedTransformer API used throughout the notebook.

### Install TransformerLens

```
pip install transformer-lens
```

### Quick check

```
python -c "from transformer_lens import HookedTransformer; HookedTransformer.from_pretrained('gelu-1l')"
```

This will download the small model weights on first run.

## Setup
Minimum dependencies (the notebook installs these inline):

```
pip install torch transformer-lens datasets tqdm accelerate numpy pandas
```

Notes:
- GPU is optional but speeds up ablation runs.
- Internet access is required to download model weights the first time.

## Running the notebook
1. Open `team5_nguyenmatthew_1258025_15148875_Neuron_Disambiguation.ipynb` in Jupyter or Colab.
2. Run cells in order.
3. Adjust `LAYER` and `NEURON` if you want to inspect a different neuron.

## Key parameters to tweak
- `LAYER`, `NEURON`: which neuron to analyze.
- `PROMPT_A`, `PROMPT_B`: contexts for cancellation/ablation tests.
- `TOP_K`, `TOP_CANCELERS`: reporting and ablation sizes.

## Outputs
The notebook prints:
- Activation maxima and context windows.
- Top boosted/suppressed tokens for the target neuron.
- Co-activating neuron IDs.
- Logit comparisons before/after ablation.

