from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd

# Load model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Replace this with your actual cluster word list (Cluster 8 for example)
cluster_words = ['yp', 'ykeeal', 'ychockh', 'yteyt', 'ychear', 'ykeoesh', 'ytchal', 'yckhod', 'ykor', 'ykeeepol']

# High-frequency reference words from various languages
reference_words = {
    'English': ['the', 'and', 'of', 'to', 'in'],
    'Latin': ['et', 'est', 'non', 'qui', 'ad'],
    'Arabic': ['ال', 'و', 'من', 'إلى', 'في'],
    'Hebrew': ['של', 'ו', 'על', 'עם', 'זה'],
    'Nahuatl': ['inin', 'tlatl', 'ca', 'noch', 'tlali']
}

# Encode all cluster words
cluster_embeddings = model.encode(cluster_words)

# Compare to each language
results = []
for lang, words in reference_words.items():
    ref_embeddings = model.encode(words)
    sim_matrix = util.cos_sim(cluster_embeddings, ref_embeddings)
    for i, word in enumerate(cluster_words):
        avg_sim = float(sim_matrix[i].numpy().mean())
        results.append({'Cluster Word': word, 'Language': lang, 'Average Cosine Similarity': avg_sim})

# Display top results per language
df = pd.DataFrame(results)
top_matches = df.sort_values(by='Average Cosine Similarity', ascending=False).groupby('Language').head(10)

print(top_matches)
