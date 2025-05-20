# lexicon_builder.py

import csv
import ast
import json
import pandas as pd
from collections import defaultdict, Counter

# === Configuration ===
CLUSTER_LOOKUP_PATH = "./data/stripped_cluster_lookup.json"
LINE_CLUSTERS_PATH = "./data/voynich_line_clusters.csv"
OUTPUT_PATH = "./results/lexicon_candidates.csv"

# === Folio category mapping
folio_categories = {
    "Botanical": range(1, 67),
    "Astronomical": range(67, 75),
    "Biological": range(75, 85),
    "Cosmological": range(85, 88),
    "Pharmaceutical": range(88, 103),
    "Unknown": range(103, 117)
}

def assign_category(folio):
    try:
        folio_num = int(''.join(filter(str.isdigit, folio)))
        for category, r in folio_categories.items():
            if folio_num in r:
                return category
    except:
        pass
    return "Uncategorized"

# === POS roles inferred from earlier clustering work
role_map = {
    8: "Function",
    3: "Root",
    6: "Object?",
    5: "Modifier?",
    0: "Object?"
}

# === Load cluster lookup
with open(CLUSTER_LOOKUP_PATH, 'r') as f:
    cluster_lookup = json.load(f)

# === Reverse map: cluster → words
reverse_cluster_map = defaultdict(list)
for word, cluster in cluster_lookup.items():
    reverse_cluster_map[cluster].append(word)

# === Build word frequencies by section
section_word_data = defaultdict(lambda: defaultdict(int))
word_sections = defaultdict(set)

with open(LINE_CLUSTERS_PATH, 'r', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        tag = row["Line Tag"]
        folio = tag.split('.')[0].replace("<", "").replace(">", "")
        section = assign_category(folio)

        try:
            clusters = ast.literal_eval(row["Cluster Sequence"])
            clusters = [c for c in clusters if c is not None]
        except:
            continue

        for cluster in clusters:
            words = reverse_cluster_map.get(cluster, [])
            for word in words:
                section_word_data[section][word] += 1
                word_sections[word].add(section)

# === Build lexicon candidate table
records = []
for section, word_counter in section_word_data.items():
    for word, freq in word_counter.items():
        cluster = cluster_lookup.get(word, None)
        pos = role_map.get(cluster, f"Cluster {cluster}")
        sections_used = sorted(word_sections[word])
        exclusive = len(sections_used) == 1

        records.append({
            "Section": section,
            "Word": word,
            "Estimated Frequency": freq,
            "Cluster": cluster,
            "POS Role": pos,
            "Section Count": len(sections_used),
            "Section Exclusive": exclusive,
            "Sections Used In": ", ".join(sections_used)
        })

# === Save results
df = pd.DataFrame(records)
df.sort_values(by=["Section", "Estimated Frequency"], ascending=[True, False], inplace=True)
df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Lexicon candidate table saved to: {OUTPUT_PATH}")
