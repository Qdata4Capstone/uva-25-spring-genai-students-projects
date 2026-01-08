# claims.py
from models import nlp, flan_pipe

def extract_claims(text: str) -> list[str]:
    """
    1. Ask Flan-T5 for one factual claim per line.
    2. Split on newlines and discard blanks.
    3. Use spaCy to break each line into atomic sentences.
    """
    prompt = (
        "Extract all factual claims from the following text. "
        "Return one claim per line without numbering:\n\n"
        f"{text}\n\nClaims:"
    )
    out   = flan_pipe(prompt)[0]["generated_text"]
    lines = [ln.strip() for ln in out.split("\n") if ln.strip()]

    claims: list[str] = []
    for ln in lines:
        doc = nlp(ln)
        for sent in doc.sents:
            claims.append(sent.text.strip())
    return claims
