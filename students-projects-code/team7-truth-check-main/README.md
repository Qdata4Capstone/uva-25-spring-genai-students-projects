# Truth-Check: Claim Extraction + Evidence + Verification

This project extracts factual claims from text, retrieves evidence from Wikipedia, and verifies each claim using an MNLI model. It also includes a Gradio UI and a FEVER evaluation script.

## Folder map
- main.py: End-to-end CLI demo (extract -> retrieve -> verify) on a sample paragraph.
- gradio_app.py: Interactive UI for claim extraction and verification.
- claims.py: Claim extraction using Flan-T5 + spaCy sentence splitting.
- evidence.py: Wikipedia retrieval using named-entity queries.
- verify.py: MNLI-based verifier that returns Supported/Contradicted/Not Verifiable.
- models.py: Loads spaCy, Flan-T5, and BART MNLI models.
- evaluate_fever.py: Runs the pipeline on the FEVER dataset and prints accuracy + confusion matrix.
- test_nli.py: Small MNLI sanity check on two Eiffel Tower claims.
- visualize.py: Plots a sample confusion matrix + class-wise recall (from hardcoded numbers).
- Class_wise_recall.png, Confusion_Matrix.png: Example plots.

## Setup
Install Python dependencies:

```
pip install -r requirements.txt
```

## Important library: spaCy
The pipeline relies on spaCy for sentence splitting and entity extraction. You must download the English model once:

```
python -m spacy download en_core_web_sm
```

## Model downloads (Transformers)
The first run will download these models from Hugging Face:
- `google/flan-t5-base` (claim extraction)
- `facebook/bart-large-mnli` (verification)

Make sure your environment has internet access for the initial download.

## Run the demo
Quick end-to-end run on a sample paragraph:

```
python main.py
```

## Launch the Gradio app

```
python gradio_app.py
```

## Evaluate on FEVER
Run evaluation on a split (optionally sample a smaller subset):

```
python evaluate_fever.py --split labelled_dev --max_samples 200
```

## Optional scripts
- Test MNLI scoring:

```
python test_nli.py
```

- Render the sample plots in `visualize.py`:

```
python visualize.py
```

## Notes and gotchas
- Wikipedia retrieval requires network access.
- The FEVER dataset downloads on first use via `datasets` and can take time.
- `visualize.py` uses hardcoded counts (it does not read outputs from `evaluate_fever.py`).

