# verify.py

import torch
import logging
from models import mnli_model, mnli_tokenizer, nlp

# Labels we’ll output
LABELS = ["Supported", "Contradicted", "Not Verifiable"]

# Find which index in logits corresponds to ENTAILMENT
ID2LABEL = mnli_model.config.id2label  # e.g. {0:"CONTRADICTION",1:"NEUTRAL",2:"ENTAILMENT"}
ENTAIL_IDX = next(i for i,l in ID2LABEL.items() if l.upper()=="ENTAILMENT")

# Confidence threshold under which we abstain
DEFAULT_THRESHOLD = 0.4
CONTRA_THRESHOLD = 0.2

def verify_claim(claim: str, evidence: str, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """
    Enhanced verifier:
      1. Split evidence into sentences via spaCy.
      2. Build hypotheses for Supported & Contradicted per sentence.
      3. Single batched MNLI call.
      4. Take the max entailment score per label across sentences.
      5. Apply threshold to decide or abstain.
    Returns dict with {'label': ..., 'confidence': ...}.
    """
    # If there's no real evidence, skip right to “Not Verifiable”
    if evidence == "No evidence found":
        return {"label": "Not Verifiable", "confidence": 0.0}

    try:
        # 1) Sentence‐split the evidence
        doc_sents = [sent.text.strip() for sent in nlp(evidence).sents if sent.text.strip()]
        if not doc_sents:
            return {"label": "Not Verifiable", "confidence": 0.0}

        # 2) Build hypothesis/premise pairs:
        #    For each sentence, we create two entries: one claiming Supported, one Contradicted.
        hypos = []
        premises = []
        for sent in doc_sents:
            for lab in ["Supported", "Contradicted"]:
                hypos.append(f"This claim is {lab} based on the evidence: {sent}")
                premises.append(claim)

        # 3) Tokenize + batch‐forward
        inputs = mnli_tokenizer(
            hypos,
            premises,
            return_tensors="pt",
            truncation=True,
            padding=True
        )
        with torch.no_grad():
            logits = mnli_model(**inputs).logits

        # 4) Softmax to get entailment probabilities
        probs = torch.softmax(logits, dim=1)[:, ENTAIL_IDX]  # shape = (2 * num_sents,)
        num_sents = len(doc_sents)

        # reshape to [num_sents x 2] → column 0=Supported, 1=Contradicted
        scores = probs.reshape(num_sents, 2)
        sup_scores = scores[:,0]
        con_scores = scores[:,1]

        # 5) Aggregate: take the max score across all sentences
        best_sup = float(sup_scores.max())
        best_con = float(con_scores.max())

        # Decide final label with thresholding
        if best_sup >= threshold and best_sup >= best_con:
            return {"label": "Supported",    "confidence": best_sup}
        if best_con >= CONTRA_THRESHOLD and best_con > best_sup:
            return {"label": "Contradicted", "confidence": best_con}

        # Otherwise abstain
        return {"label": "Not Verifiable", "confidence": max(best_sup, best_con)}

    except Exception as e:
        logging.exception(f"Verification error for claim: {claim}")
        return {"label": "Error", "confidence": 0.0, "error": str(e)}
