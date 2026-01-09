# Team 8 - LLM for Paper Review + Evaluation

This folder contains a single notebook that generates structured academic reviews of PDF papers using an LLM, exports reviews to PDF/DOCX, and evaluates summary quality with ROUGE/BLEU/BERTScore/SBERT metrics.

## Folder contents
- team8_paper_review_analysis.ipynb: End-to-end notebook for review generation and evaluation.

## What the notebook does
1. Installs dependencies (OpenAI SDK, PDF parsing, formatting, metrics).
2. Reads PDF papers and extracts text.
3. Generates a structured JSON review with an LLM (title, summary, strengths/weaknesses, ratings).
4. Formats the review into PDF or DOCX.
5. Creates baseline reviews for comparison.
6. Computes ROUGE, BLEU, BERTScore, and SBERT cosine similarity against paper abstracts.

## Important library: OpenAI Python SDK
The notebook uses `openai.OpenAI` to call the LLM for review generation.

### Install

```
pip install openai
```

### API key setup
The notebook currently uses `openai.OpenAI(api_key='your-key')`. Replace with a real key or set an environment variable:

```
export OPENAI_API_KEY=... 
```

Then initialize:

```
from openai import OpenAI
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
```

## Setup
Minimum dependencies (also installed in the notebook):

```
pip install openai pydantic python-docx reportlab tiktoken PyPDF2
```

Evaluation dependencies:

```
pip install rouge-score nltk bert-score sentence-transformers
```

Optional PDF output helper:

```
pip install fpdf
```

## Running the notebook
1. Place your PDFs in a `papers/` folder (the notebook expects this path).
2. Open `team8_paper_review_analysis.ipynb` in Jupyter or Colab.
3. Run cells in order to generate reviews and metrics.

## Expected inputs and outputs
Inputs:
- `papers/` directory with `*.pdf` files.

Outputs:
- `output/` (or `outputs/`) directory with `review_<paper_id>.pdf`
- `baseline_outputs/` directory with baseline review PDFs
- Printed DataFrames with ROUGE/BLEU/BERTScore/SBERT metrics

## Notes and gotchas
- PDF text extraction quality varies by paper; scanned PDFs may fail.
- The notebook uses multiple versions of similar evaluation cells; pick one path and stick to it.
- Large papers may exceed token limits; the `PaperProcessor` splits by token count.

