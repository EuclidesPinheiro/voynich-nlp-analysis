# --- cluster_roots_AB.py ---
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
from docx import Document # Adicionado para ler o .docx

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

# def load_words_from_directory(directory): # NÃO MAIS USADO DIRETAMENTE PARA VOCABULÁRIO PRINCIPAL
#     """Lê todas as palavras de arquivos .txt no diretório."""
#     words = []
#     for filename in os.listdir(directory):
#         if filename.endswith('.txt'):
#             filepath = os.path.join(directory, filename)
#             with open(filepath, 'r', encoding='utf-8') as f:
#                 words.extend(f.read().split())
#     return words

### NOVA FUNÇÃO ###
def load_words_from_docx(docx_filepath):
    """Lê todas as palavras do arquivo .docx, ignorando tags de linha."""
    logging.info(f"Carregando palavras do arquivo DOCX: {docx_filepath}")
    if not os.path.exists(docx_filepath):
        logging.error(f"Arquivo DOCX não encontrado em: {docx_filepath}")
        return []
        
    document = Document(docx_filepath)
    all_words_from_docx = []
    for para_idx, para in enumerate(document.paragraphs):
        text = para.text.strip()
        if not text: # Pula parágrafos vazios
            continue
        
        parts = text.split()
        # Ignora a primeira parte se for uma tag (começa com '<' e termina com '>')
        # ou se o parágrafo não tiver palavras após a tag
        if parts:
            if parts[0].startswith('<') and parts[0].endswith('>'):
                if len(parts) > 1:
                    all_words_from_docx.extend(parts[1:])
            else: # Se não começar com tag, considera todas as partes como palavras
                all_words_from_docx.extend(parts)
    logging.info(f"Extraídas {len(all_words_from_docx)} palavras (tokens) do DOCX.")
    return all_words_from_docx

def save_json(data, filepath):
    """Salva dados em formato JSON."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def main(args):
    output_suffix = args.output_suffix or ("no_stripping" if args.disable_suffix_stripping else "")
    if args.output_suffix and args.output_suffix.startswith('_'): # Normaliza o sufixo
        output_suffix = args.output_suffix[1:]

    ### ALTERAÇÃO PRINCIPAL: Carregar palavras do DOCX ###
    logging.info(f"Processando palavras do arquivo: {args.docx_path}")
    # words = load_words_from_directory(args.data_dir) # Linha antiga
    words = load_words_from_docx(args.docx_path) # Nova linha
    logging.info(f"Total de palavras (tokens) carregadas do DOCX: {len(words)}")

    if not words:
        logging.error("Nenhuma palavra carregada do DOCX. Verifique o caminho e o conteúdo do arquivo. Encerrando.")
        return

    processed_words = [strip_suffix(word, args.suffix_list, args.disable_suffix_stripping) for word in words]
    unique_words = list(set(processed_words))
    logging.info(f"Palavras únicas após processamento: {len(unique_words)}")

    if "" in unique_words: # Adicionando uma verificação para strings vazias
        logging.warning("Uma string vazia foi encontrada entre as palavras únicas e será removida.")
        unique_words = [word for word in unique_words if word] # Remove strings vazias

    if not unique_words:
        logging.error("Nenhuma palavra única encontrada após a remoção de strings vazias. Encerrando.")
        return

    logging.info(f"Carregando modelo SBERT: {args.sbert_model}")
    model = SentenceTransformer(args.sbert_model)
    logging.info(f"Encoding {len(unique_words)} palavras únicas...") # Log atualizado
    embeddings = model.encode(unique_words, show_progress_bar=True)

    logging.info(f"Aplicando KMeans com {args.n_clusters} clusters")
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(embeddings)

    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10')
    title = "Voynich Clusters (from AB.docx)" # Título atualizado
    title_status = " (No Suffix Stripping)" if args.disable_suffix_stripping else " (Suffix Stripped)"
    title_custom_suffix = f" - {output_suffix.replace('_',' ')}" if output_suffix else ""
    plt.title(title + title_status + title_custom_suffix)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    
    plot_path = generate_output_path(args.output_dir, args.plot_output_path_base, output_suffix) # Correção: usar args.plot_output_path_base
    plt.savefig(plot_path)
    logging.info(f"Gráfico salvo em: {plot_path}")

    # Salvar JSONs
    cluster_map = {word: int(label) for word, label in zip(unique_words, labels)}
    clustered_words_dict = {i: [] for i in range(args.n_clusters)} # Renomeado para evitar conflito
    for word, label in cluster_map.items():
        clustered_words_dict[label].append(word)

    logging.info("Exibindo até 10 palavras por cluster:")
    for cluster_id, words_in_cluster in clustered_words_dict.items(): # Usar a variável correta
        logging.info(f"Cluster {cluster_id}: {words_in_cluster[:10]}")

    unique_words_path = generate_output_path(args.output_dir, args.unique_words_output_base, output_suffix)
    save_json(unique_words, unique_words_path)
    logging.info(f"Palavras únicas salvas em: {unique_words_path}")

    cluster_map_path = generate_output_path(args.output_dir, args.cluster_lookup_output_base, output_suffix)
    save_json(cluster_map, cluster_map_path)
    logging.info(f"Mapeamento de cluster salvo em: {cluster_map_path}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Clusteriza palavras do Voynich (lidas de um arquivo DOCX) com ou sem remoção de sufixos.")
    ### ALTERAÇÃO: --data_dir removido como fonte principal, --docx_path adicionado/enfatizado ###
    parser.add_argument("--docx_path", type=str, default="./data/AB.docx", # Default para o arquivo principal
                        help="Caminho para o arquivo .docx da transliteração do Voynich.")
    parser.add_argument("--suffix_list", nargs='+', default=['aiin', 'dy', 'in', 'chy', 'chey', 'edy', 'ey', 'y'])
    parser.add_argument("--sbert_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--n_clusters", type=int, default=10)
    parser.add_argument("--disable_suffix_stripping", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./results_AB_docx") # Default de saída alterado
    parser.add_argument("--output_suffix", type=str, default="")
    parser.add_argument("--plot_output_path_base", type=str, default="Figure_1_AB.png") # Nome base do plot alterado
    parser.add_argument("--unique_words_output_base", type=str, default="unique_words_AB.json") # Nome base alterado
    parser.add_argument("--cluster_lookup_output_base", type=str, default="cluster_lookup_AB.json") # Nome base alterado
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)