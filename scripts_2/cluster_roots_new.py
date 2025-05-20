# --- cluster_roots.py ---
import os
import re
import json
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_output_path(base_dir, filename, suffix_str=""):
    """Cria caminho de saída com sufixo e garante que o diretório existe."""
    os.makedirs(base_dir, exist_ok=True)
    base, ext = os.path.splitext(filename)
    if suffix_str and not suffix_str.startswith('_'):
        suffix_str = "_" + suffix_str
    return os.path.join(base_dir, f"{base}{suffix_str}{ext}")

def strip_suffix(word, suffixes, disable_stripping):
    """Remove o sufixo da palavra, se aplicável."""
    if disable_stripping:
        return word
    for suffix in sorted(suffixes, key=len, reverse=True):
        if word.endswith(suffix):
            return re.sub(f'{re.escape(suffix)}$', '', word)
    return word

def load_words_from_directory(directory):
    """Lê todas as palavras de arquivos .txt no diretório."""
    words = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                words.extend(f.read().split())
    return words

def save_json(data, filepath):
    """Salva dados em formato JSON."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def main(args):
    output_suffix = args.output_suffix or ("no_stripping" if args.disable_suffix_stripping else "")

    logging.info(f"Carregando palavras do diretório: {args.data_dir}")
    words = load_words_from_directory(args.data_dir)
    logging.info(f"Total de palavras carregadas: {len(words)}")

    processed_words = [strip_suffix(word, args.suffix_list, args.disable_suffix_stripping) for word in words]
    unique_words = list(set(processed_words))
    logging.info(f"Palavras únicas após processamento: {len(unique_words)}")

    if not unique_words:
        logging.error("Nenhuma palavra única encontrada. Encerrando.")
        return

    logging.info(f"Carregando modelo SBERT: {args.sbert_model}")
    model = SentenceTransformer(args.sbert_model)
    embeddings = model.encode(unique_words, show_progress_bar=True)

    logging.info(f"Aplicando KMeans com {args.n_clusters} clusters")
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(embeddings)

    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10')
    title = "Voynich Clusters"
    title += " (No Suffix Stripping)" if args.disable_suffix_stripping else " (Suffix Stripped)"
    if output_suffix:
        title += f" - {output_suffix}"
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    
    plot_path = generate_output_path(args.output_dir, os.path.basename(args.plot_output_path_base), output_suffix)
    plt.savefig(plot_path)
    logging.info(f"Gráfico salvo em: {plot_path}")

    # Salvar JSONs
    cluster_map = {word: int(label) for word, label in zip(unique_words, labels)}
    clustered_words = {i: [] for i in range(args.n_clusters)}
    for word, label in cluster_map.items():
        clustered_words[label].append(word)

    logging.info("Exibindo até 10 palavras por cluster:")
    for cluster_id, words in clustered_words.items():
        logging.info(f"Cluster {cluster_id}: {words[:10]}")

    save_json(unique_words, generate_output_path(args.output_dir, args.unique_words_output_base, output_suffix))
    save_json(cluster_map, generate_output_path(args.output_dir, args.cluster_lookup_output_base, output_suffix))
    
    unique_words_path = generate_output_path(args.output_dir, args.unique_words_output_base, output_suffix)
    save_json(unique_words, unique_words_path)
    logging.info(f"Palavras únicas salvas em: {unique_words_path}")

    cluster_map_path = generate_output_path(args.output_dir, args.cluster_lookup_output_base, output_suffix)
    save_json(cluster_map, cluster_map_path)
    logging.info(f"Mapeamento de cluster salvo em: {cluster_map_path}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Clusteriza palavras do Voynich com ou sem remoção de sufixos.")
    parser.add_argument("--data_dir", type=str, default="./data/voynitchese")
    parser.add_argument("--suffix_list", nargs='+', default=['aiin', 'dy', 'in', 'chy', 'chey', 'edy', 'ey', 'y'])
    parser.add_argument("--sbert_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--n_clusters", type=int, default=10)
    parser.add_argument("--disable_suffix_stripping", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./results_custom")
    parser.add_argument("--output_suffix", type=str, default="")
    parser.add_argument("--plot_output_path_base", type=str, default="Figure_1.png")
    parser.add_argument("--unique_words_output_base", type=str, default="unique_stripped_words.json")
    parser.add_argument("--cluster_lookup_output_base", type=str, default="stripped_cluster_lookup.json")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
