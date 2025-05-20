# pos_model.py

import csv
import ast
from collections import defaultdict, Counter
import pandas as pd

# === Configuration ===
CLUSTER_LOOKUP_PATH = "./data/stripped_cluster_lookup.json"
LINE_CLUSTERS_PATH = "./data/voynich_line_clusters.csv"

# === Role Inference Heuristics ===
def infer_role(total, unique_words, starts, ends):
    if total > 3000 and unique_words < 500:
        return "Function"
    elif unique_words > 1000:
        return "Root"
    elif starts > 500 and ends > 500:
        return "Modifier"
    elif starts > ends:
        return "Subject Marker?"
    elif ends > starts:
        return "Object Marker?"
    else:
        return "Unknown"

# === Load cluster-word mapping
import json
with open(CLUSTER_LOOKUP_PATH, "r") as f:
    cluster_lookup = json.load(f)

reverse_cluster_map = defaultdict(list)
for word, cluster in cluster_lookup.items():
    reverse_cluster_map[cluster].append(word)

# === Load line cluster usage
line_starts = Counter()
line_ends = Counter()
cluster_frequency = Counter()

with open(LINE_CLUSTERS_PATH, 'r', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            clusters = ast.literal_eval(row["Cluster Sequence"])
            clusters = [c for c in clusters if c is not None]
        except:
            continue
        if clusters:
            line_starts[clusters[0]] += 1
            line_ends[clusters[-1]] += 1
        for cluster in clusters:
            cluster_frequency[cluster] += 1

# === Summarize cluster roles
summary = []
for cluster in sorted(reverse_cluster_map.keys()):
    words = reverse_cluster_map[cluster]
    unique = len(set(words))
    total = cluster_frequency[cluster]
    starts = line_starts[cluster]
    ends = line_ends[cluster]
    summary.append({
        "Cluster": cluster,
        "Total Occurrences": total,
        "Unique Words": unique,
        "Line Starts": starts,
        "Line Ends": ends,
        "Inferred Role": infer_role(total, unique, starts, ends)
    })

# === Output summary
df_summary = pd.DataFrame(summary)
df_summary.to_csv("./results/cluster_role_summary.csv", index=False)
print("Saved: ./results/cluster_role_summary.csv")
