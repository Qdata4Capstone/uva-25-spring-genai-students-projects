# Feed in a paragraph to multilingual RoBERTa 
# spits back a sentiment score

from transformers import pipeline
import pandas as pd
import numpy as np
from collections import Counter

# Load the model just once
sentPipe = pipeline(
    task="sentiment-analysis",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
    tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment",
    truncation=True,
    padding=True,
)

# model spits LABEL_0/1/2 – map that to words we can actually understand
LABEL2STR = {
    # old-style keys
    "LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive",
    # new-style keys (Transformers ≥4.41)
    "negative": "negative", "neutral": "neutral", "positive": "positive",
}

def classLabel(pred):
    # returns "negative" / "neutral" / "positive"
    return LABEL2STR[pred["label"]]

def signedScore(pred):
    # interpr model output
    # score = pred["score"]
    # if pred["label"].endswith("0"):
    #     return -score
    # if pred["label"].endswith("2"):
    #     return score
    # return 0.0 
    score = pred["score"]
    label = pred["label"]

    # new-style tags
    if label == "negative":
        return -score
    if label == "positive":
        return  score

    # old-style fallback
    if label.endswith("0"):
        return -score
    if label.endswith("2"):
        return  score

    # must be neutral
    return 0.0

# def classLabel(pred):
#     return LABEL2STR[pred["label"]]

def majorityVote(labels):
    counts = Counter(labels)
    if counts.most_common(1)[0][1] >= 3:
        return counts.most_common(1)[0][0]

    mapping = {"negative": -1, "neutral": 0, "positive": 1}
    if np.mean([mapping[l] for l in labels]) > 0.33:
        return "positive"
    if np.mean([mapping[l] for l in labels]) < -0.33:
        return "negative"
    return "neutral" 

# grade ONE paragraph against its reference bundle
def evaluateTopic(generated, references):
    # model run for generated text
    genRaw   = sentPipe(generated[:512])[0]
    genScore = signedScore(genRaw)
    genClass = classLabel(genRaw)

    # model run for references
    refRaw    = sentPipe(references)
    refScores = [signedScore(r) for r in refRaw]
    refLabels = [classLabel(r) for r in refRaw]
    refMean   = float(np.mean(refScores))
    refClass  = majorityVote(refLabels)

    return {
        "generated_polarity": genScore,
        "reference_mean":     refMean,
        "difference":         genScore - refMean,
        "abs_difference":     abs(genScore - refMean),
        "generated_class":    genClass,
        "reference_class":    refClass,
        "correct":            genClass == refClass,
    }

# do it for a whole table in one shot
def evaluateDataFrame(df):
    rows, gold, pred = [], [], []
    for _, row in df.iterrows():
        m = evaluateTopic(row.generated, row.references)
        rows.append({**row, **m})
        gold.append(m["reference_class"])
        pred.append(m["generated_class"])

    out = pd.DataFrame(rows)
    acc = out.correct.mean()

    cm = pd.crosstab(
        pd.Series(gold, name="gold"),
        pd.Series(pred, name="pred"),
        dropna=False
    ).reindex(index=["negative","neutral","positive"],
              columns=["negative","neutral","positive"]).fillna(0).astype(int)

    return out, acc, cm
