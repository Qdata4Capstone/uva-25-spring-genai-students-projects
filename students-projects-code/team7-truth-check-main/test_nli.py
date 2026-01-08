# test_nli.py

import torch
from models import mnli_model, mnli_tokenizer

# find which logit index = ENTAILMENT
id2label   = mnli_model.config.id2label
entail_idx = next(i for i,l in id2label.items() if l.upper()=="ENTAILMENT")

samples = [
    ("The Eiffel Tower is in Paris.",
     "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France."),
    ("The Eiffel Tower is in Berlin.",
     "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.")
]

for claim, evidence in samples:
    hypos    = [
        f"This claim is Supported based on the evidence: {evidence}",
        f"This claim is Contradicted based on the evidence: {evidence}"
    ]
    premises = [claim, claim]

    inputs = mnli_tokenizer(
        hypos,
        premises,
        return_tensors="pt",
        truncation=True,
        padding=True
    )
    with torch.no_grad():
        logits = mnli_model(**inputs).logits

    probs = torch.softmax(logits, dim=1)[:, entail_idx]
    print(f"\nClaim: {claim!r}")
    print(f"  Supported score:    {probs[0].item():.3f}")
    print(f"  Contradicted score: {probs[1].item():.3f}")
