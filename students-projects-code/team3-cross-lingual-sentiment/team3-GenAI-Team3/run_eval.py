from pathlib import Path
import pandas as pd
import seaborn as sns, matplotlib.pyplot as plt
from sentiment_eval import evaluateDataFrame

# USE THESE FOR SMALLER EXCERPTS
# ground‑truth refs
# from ground_truth_en import gt_df
# from ground_truth_fr import df_verite
# from ground_truth_ja import gt_df_ja_ref
# from ground_truth_zh import gt_df_zh_ref

# USE THESE FOR LARGER EXCERPTS
from english import gt_df
from french import gt_df_fr
from japanese import gt_df_ja
from chinese import gt_df_zh

# generated paragraphs from Shunqiang's code to evaluate
from gen_dicts import gen_en, gen_fr, gen_ja, gen_zh

def injectGenerated(ref_df, gen_dict):
    # Replace the blank ‘generated’ column with our generated text
    return ref_df.drop(columns="generated").merge(
        pd.DataFrame({"topic": list(gen_dict), "generated": list(gen_dict.values())}),
        on="topic")

# datasets = {
#     "English":  injectGenerated(gt_df,         gen_en),
#     "French":   injectGenerated(df_verite,     gen_fr),
#     "Japanese": injectGenerated(gt_df_ja_ref,  gen_ja),
#     "Chinese":  injectGenerated(gt_df_zh_ref,  gen_zh),
# }

datasets = {
    "English":  injectGenerated(gt_df,     gen_en),
    "French":   injectGenerated(gt_df_fr,  gen_fr),
    "Japanese": injectGenerated(gt_df_ja,  gen_ja),
    "Chinese":  injectGenerated(gt_df_zh,  gen_zh),
}

outDir = Path("results")
outDir.mkdir(exist_ok=True)

bigTable = []
accs = {}

for lang, df in datasets.items():
    res, acc, cm = evaluateDataFrame(df)
    accs[lang] = acc*100
    bigTable.append(res)

    print(f"\n▶ {lang}: {acc:.0%} accurate")
    print(res[["topic","generated_class","reference_class",
               "generated_polarity","reference_mean","difference"]].to_string(index=False))

    # save heat‑map
    sns.heatmap(cm, annot=True, fmt="d", cmap="rocket_r")
    plt.title(f"Confusion – {lang}")
    plt.tight_layout()
    plt.savefig(outDir / f"conf_{lang}.png", dpi=300)
    plt.close()

# bar chart of the accuracy scores
sns.barplot(x=list(accs.keys()), y=list(accs.values()), palette="pastel")
plt.ylabel("Accuracy (%)")
plt.ylim(0,100)
plt.title("Sentiment alignment acc")
plt.tight_layout()
plt.savefig(outDir / "accuracy_bar.png", dpi=300)
plt.close()

pd.concat(bigTable).to_csv(outDir / "per_topic_metrics.csv", index=False)
print("Saved to", outDir.resolve())
