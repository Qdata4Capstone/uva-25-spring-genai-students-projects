# visualize.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Build the confusion matrix DataFrame (from your 200-sample example) ---
data = {
    'Supported':       [40, 16, 10],
    'Contradicted':    [20, 48, 10],
    'Not Verifiable':  [20, 16, 20]
}
index = ['Supported', 'Contradicted', 'Not Verifiable']
cm = pd.DataFrame(data, index=index)

# 1) Print it
print("\nConfusion Matrix (rows=true, cols=predicted):\n")
print(cm)

# 2) Plot as heatmap
fig, ax = plt.subplots(figsize=(6,6))
im = ax.imshow(cm.values, aspect='equal')
ax.set_xticks(range(len(cm.columns)))
ax.set_xticklabels(cm.columns, rotation=45, ha='right')
ax.set_yticks(range(len(cm.index)))
ax.set_yticklabels(cm.index)
ax.set_title('Confusion Matrix Heatmap')
for i in range(len(cm.index)):
    for j in range(len(cm.columns)):
        ax.text(j, i, cm.values[i, j], ha='center', va='center', color='white' if im.norm(cm.values[i, j])>0.5 else 'black')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

# 3) Compute & plot recall per class
# True counts per class: Supported=80, Contradicted=80, Not Verifiable=40
true_counts = np.array([80, 80, 40])
recall = cm.values.diagonal() / true_counts

fig, ax = plt.subplots(figsize=(6,4))
ax.bar(cm.index, recall)
ax.set_ylim(0, 1)
ax.set_ylabel('Recall')
ax.set_title('Class-wise Recall')
for i, v in enumerate(recall):
    ax.text(i, v + 0.02, f"{v:.1%}", ha='center')
plt.tight_layout()
plt.show()
