# --- pos_model_modificado.py ---

import csv
import ast
import json # Movido para o topo com outros imports
import argparse
import logging
from collections import defaultdict, Counter
import pandas as pd
import os # Adicionado para os.path.basename

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Função para gerar caminho de saída (pode ser importada de um utils.py no futuro)
def generate_output_path(base_dir, filename, suffix_str=""):
    """Cria caminho de saída com sufixo e garante que o diretório existe."""
    os.makedirs(base_dir, exist_ok=True) # Garante que o diretório de saída existe
    base, ext = os.path.splitext(filename)
    if suffix_str and not suffix_str.startswith('_'):
        suffix_str = "_" + suffix_str
    return os.path.join(base_dir, f"{base}{suffix_str}{ext}")

# === Role Inference Heuristics ===
# (Mantida como no original, para observarmos seu comportamento)
def infer_role(total, unique_words, starts, ends):
    # Estes limiares podem precisar de ajuste ao analisar dados sem remoção de sufixos
    if total > 3000 and unique_words < 500:
        return "Function"
    elif unique_words > 1000:
        return "Root"
    elif starts > 500 and ends > 500: # Originalmente > 500 para ambos
        return "Modifier"
    elif starts > (ends + 100): # Adicionado um diferencial para ser mais claro que "Subject Marker?"
        return "Subject Marker?"
    elif ends > (starts + 100): # Adicionado um diferencial para ser mais claro que "Object Marker?"
        return "Object Marker?"
    else:
        return "Unknown"

def load_cluster_lookup(filepath):
    """Carrega o mapeamento de palavra para cluster."""
    logging.info(f"Carregando mapeamento de cluster de: {filepath}")
    with open(filepath, "r", encoding='utf-8') as f:
        cluster_lookup = json.load(f)
    
    reverse_cluster_map = defaultdict(list)
    for word, cluster in cluster_lookup.items():
        reverse_cluster_map[cluster].append(word)
    logging.info(f"Mapeamento de cluster carregado. {len(reverse_cluster_map)} clusters encontrados.")
    return cluster_lookup, reverse_cluster_map

def process_line_clusters(filepath, reverse_cluster_map):
    """Processa o arquivo de sequências de cluster por linha para obter estatísticas."""
    logging.info(f"Processando sequências de cluster por linha de: {filepath}")
    line_starts = Counter()
    line_ends = Counter()
    cluster_frequency = Counter()
    
    # Verificar se todos os clusters em reverse_cluster_map são inteiros
    # Isso é importante porque as chaves em reverse_cluster_map vêm do JSON e podem ser strings
    # enquanto os clusters no CSV, após ast.literal_eval, serão inteiros.
    # Garantir que estamos comparando tipos iguais.
    int_reverse_cluster_map = {int(k): v for k, v in reverse_cluster_map.items()}


    processed_lines = 0
    with open(filepath, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            try:
                # ast.literal_eval pode ser lento para arquivos grandes.
                # Se o formato do CSV for sempre uma lista de inteiros (ex: "[1, 2, 3]"),
                # um parser mais simples poderia ser usado.
                clusters_str = row.get("Cluster Sequence", "[]") # Default para string vazia se coluna não existir
                clusters = ast.literal_eval(clusters_str)
                # Garantir que todos os elementos são inteiros e não None
                clusters = [int(c) for c in clusters if c is not None] 
            except (ValueError, SyntaxError) as e:
                logging.warning(f"Erro ao processar a sequência de cluster na linha {row_idx+1} do CSV: '{row.get('Cluster Sequence', '')}'. Erro: {e}. Pulando linha.")
                continue
            
            if clusters:
                line_starts[clusters[0]] += 1
                line_ends[clusters[-1]] += 1
            for cluster_id in clusters:
                cluster_frequency[cluster_id] += 1
            processed_lines +=1
    
    logging.info(f"Processadas {processed_lines} linhas do arquivo de sequências de cluster.")
    # Validar se todos os clusters encontrados no CSV existem no reverse_map
    # for c_id in list(line_starts.keys()) + list(line_ends.keys()) + list(cluster_frequency.keys()):
    #     if c_id not in int_reverse_cluster_map:
    #         logging.warning(f"Cluster ID {c_id} encontrado no arquivo CSV mas não no cluster_lookup.json. "
    #                         "Pode indicar uma inconsistência ou um cluster vazio não representado.")

    return line_starts, line_ends, cluster_frequency, int_reverse_cluster_map


def summarize_cluster_roles(int_reverse_cluster_map, cluster_frequency, line_starts, line_ends):
    """Gera o sumário de papéis para cada cluster."""
    logging.info("Gerando sumário de papéis dos clusters...")
    summary = []
    # Iterar sobre os clusters que realmente existem (têm palavras associadas)
    # e também os que apareceram nas sequências de linhas (para o caso de clusters vazios no lookup mas presentes nos dados)
    all_present_cluster_ids = set(int_reverse_cluster_map.keys()) | set(cluster_frequency.keys()) | set(line_starts.keys()) | set(line_ends.keys())

    for cluster_id in sorted(list(all_present_cluster_ids)):
        words_in_cluster = int_reverse_cluster_map.get(cluster_id, []) # Pega palavras do cluster, ou lista vazia se não houver
        unique_count = len(set(words_in_cluster))
        total_occurrences = cluster_frequency.get(cluster_id, 0) # Pega frequência, ou 0 se não ocorrer
        starts_count = line_starts.get(cluster_id, 0)
        ends_count = line_ends.get(cluster_id, 0)
        
        role = infer_role(total_occurrences, unique_count, starts_count, ends_count)
        
        summary.append({
            "Cluster": cluster_id,
            "Total Occurrences": total_occurrences,
            "Unique Words": unique_count,
            "Line Starts": starts_count,
            "Line Ends": ends_count,
            "Inferred Role": role,
            "Sample Words": words_in_cluster[:5] # Adiciona algumas palavras de exemplo
        })
    logging.info("Sumário de papéis dos clusters gerado.")
    return pd.DataFrame(summary)

def main(args):
    # Construir caminhos de entrada
    cluster_lookup_path = generate_output_path(args.input_dir, os.path.basename(args.cluster_lookup_base), args.input_suffix)
    line_clusters_path = generate_output_path(args.input_dir, os.path.basename(args.line_clusters_base), args.input_suffix)

    # Carregar dados
    _, reverse_cluster_map = load_cluster_lookup(cluster_lookup_path) # Ignora o cluster_lookup direto
    line_starts, line_ends, cluster_frequency, int_reverse_cluster_map = process_line_clusters(line_clusters_path, reverse_cluster_map)

    # Gerar e salvar sumário
    df_summary = summarize_cluster_roles(int_reverse_cluster_map, cluster_frequency, line_starts, line_ends)
    
    output_summary_path = generate_output_path(args.output_dir, os.path.basename(args.output_summary_base), args.output_suffix)
    df_summary.to_csv(output_summary_path, index=False)
    logging.info(f"Sumário de papéis dos clusters salvo em: {output_summary_path}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Infere papéis para clusters de palavras do Voynich.")
    
    # Argumentos de Entrada
    parser.add_argument("--input_dir", type=str, default="./results_custom", # Ajuste o default se necessário
                        help="Diretório onde os arquivos de entrada (lookup e line_clusters) estão localizados.")
    parser.add_argument("--input_suffix", type=str, default="",
                        help="Sufixo usado nos nomes dos arquivos de entrada (ex: '_no_strip').")
    parser.add_argument("--cluster_lookup_base", type=str, default="stripped_cluster_lookup.json",
                        help="Nome base do arquivo JSON de mapeamento palavra-cluster.")
    parser.add_argument("--line_clusters_base", type=str, default="voynich_line_clusters.csv",
                        help="Nome base do arquivo CSV com sequências de cluster por linha.")

    # Argumentos de Saída
    parser.add_argument("--output_dir", type=str, default="./results_custom", # Ajuste o default se necessário
                        help="Diretório para salvar o arquivo de sumário de saída.")
    parser.add_argument("--output_suffix", type=str, default="",
                        help="Sufixo a ser adicionado ao nome do arquivo de sumário de saída.")
    parser.add_argument("--output_summary_base", type=str, default="cluster_role_summary.csv",
                        help="Nome base para o arquivo CSV de sumário de papéis dos clusters.")
    
    # Outros (se necessário no futuro, como limiares para infer_role)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Lógica para garantir que input_suffix e output_suffix não comecem com '_' se fornecidos assim
    if args.input_suffix and args.input_suffix.startswith('_'):
        args.input_suffix = args.input_suffix[1:]
    if args.output_suffix and args.output_suffix.startswith('_'):
        args.output_suffix = args.output_suffix[1:]
        
    main(args)