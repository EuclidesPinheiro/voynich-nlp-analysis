# transition_matrix.py

import csv
import ast
from collections import Counter
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === Configuration ===
LINE_CLUSTERS_PATH = "./data/voynich_line_clusters.csv"
OUTPUT_CSV = "./results/cluster_transition_matrix.csv"
PLOT_PATH = "./results/transition_matrix_heatmap.png"

# === Load transitions
transitions = Counter()
totals = Counter()

with open(LINE_CLUSTERS_PATH, "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            clusters = ast.literal_eval(row["Cluster Sequence"])
            clusters = [c for c in clusters if c is not None]
        except:
            continue

        for i in range(len(clusters) - 1):
            transitions[(clusters[i], clusters[i + 1])] += 1
            totals[clusters[i]] += 1

# === Build transition matrix
unique_clusters = sorted(set([c for pair in transitions for c in pair]))
index_map = {c: i for i, c in enumerate(unique_clusters)}
matrix = np.zeros((len(unique_clusters), len(unique_clusters)))

for (c1, c2), count in transitions.items():
    i, j = index_map[c1], index_map[c2]
    matrix[i][j] = count

# Normalize to probability matrix
row_sums = matrix.sum(axis=1, keepdims=True)
prob_matrix = np.divide(matrix, row_sums + 1e-9)

# Save to CSV
df = pd.DataFrame(prob_matrix, index=unique_clusters, columns=unique_clusters)
df.to_csv(OUTPUT_CSV)
print(f"Saved transition matrix: {OUTPUT_CSV}")

# Optional: Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df, annot=False, cmap="Blues", xticklabels=True, yticklabels=True)
plt.title("Cluster Transition Probability Heatmap")
plt.xlabel("To Cluster")
plt.ylabel("From Cluster")
plt.tight_layout()
plt.savefig(PLOT_PATH)
print(f"Saved heatmap: {PLOT_PATH}")
