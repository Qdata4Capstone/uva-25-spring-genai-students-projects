# evaluate_fever.py

import argparse
from datasets import load_dataset
from tqdm.auto import tqdm
from evidence import get_wikipedia_evidence
from verify import verify_claim

FEVER_TO_OUR = {
    "SUPPORTS":        "Supported",
    "REFUTES":         "Contradicted",
    "NOT ENOUGH INFO": "Not Verifiable"
}

def evaluate(split="labelled_dev", max_samples=None):
    ds = load_dataset(
        "fever",
        "v1.0",
        split=split,
        trust_remote_code=True
    )

    # --- ADD THIS BLOCK TO SHUFFLE & SLICE ---
    if max_samples is not None:
        ds = ds.shuffle(seed=42).select(range(max_samples))
    # ----------------------------------------

    total, correct = 0, 0
    labels = ["Supported","Contradicted","Not Verifiable","Error"]
    confusion = {g: {p: 0 for p in labels} for g in labels}

    for ex in tqdm(ds, total=len(ds)):
        claim    = ex["claim"]
        gold_raw = ex["label"]
        gold     = FEVER_TO_OUR.get(gold_raw, "Error")

        evidence = get_wikipedia_evidence(claim)
        pred     = verify_claim(claim, evidence).get("label", "Error")

        total   += 1
        if pred == gold:
            correct += 1
        confusion[gold][pred] += 1

    acc = correct / total if total else 0.0
    print(f"\nAccuracy: {acc:.2%} ({correct}/{total})\n")
    print("Confusion matrix (rows=gold, cols=pred):")
    header = "     " + "  ".join(f"{p[:3]}" for p in labels)
    print(header)
    for g in labels:
        row = f"{g[:3]} " + "  ".join(f"{confusion[g][p]:3d}" for p in labels)
        print(row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        type=str,
        default="labelled_dev",
        choices=[
            "train", "labelled_dev", "unlabelled_dev",
            "unlabelled_test", "paper_dev", "paper_test"
        ]
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="If set, randomly sample this many examples from the split"
    )
    args = parser.parse_args()
    evaluate(split=args.split, max_samples=args.max_samples)
