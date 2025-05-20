# --- map_lines_to_clusters.py ---

import json
import os
import re
import csv
from docx import Document
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_output_path(base_dir, filename, suffix_str=""):
    """Helper function to create output path with suffix and ensure directory exists."""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        logging.info(f"Created output directory: {base_dir}")

    base, ext = os.path.splitext(filename)
    if suffix_str and not suffix_str.startswith('_'):
        suffix_str = "_" + suffix_str
    return os.path.join(base_dir, f"{base}{suffix_str}{ext}")

def main(args):
    voynich_doc_path = args.voynich_doc_path
    current_suffixes = args.suffix_list
    output_dir = args.output_dir ### ALTERAÇÃO ###
    output_file_suffix = args.output_suffix if args.output_suffix else "" ### ALTERAÇÃO ###

    if args.disable_suffix_stripping and not args.output_suffix:
        output_file_suffix = "_no_stripping"
    elif args.output_suffix and args.output_suffix.startswith('_'):
        output_file_suffix = args.output_suffix[1:]


    def strip_suffix_local(word_to_strip):
        if args.disable_suffix_stripping:
            return word_to_strip
        for suffix_val in sorted(current_suffixes, key=len, reverse=True):
            if word_to_strip.endswith(suffix_val):
                return re.sub(f'{re.escape(suffix_val)}$', '', word_to_strip)
        return word_to_strip

    # === Load stripped word → cluster ID mapping ===
    # O arquivo de lookup virá do mesmo output_dir e com o mesmo sufixo do script anterior
    cluster_lookup_base_filename = os.path.basename(args.cluster_json_path_base) # ex: stripped_cluster_lookup.json
    cluster_json_path = generate_output_path(output_dir, cluster_lookup_base_filename, output_file_suffix)
    
    logging.info(f"Attempting to load cluster lookup from: {cluster_json_path}")

    if os.path.exists(cluster_json_path):
        with open(cluster_json_path, 'r', encoding='utf-8') as f:
            cluster_lookup = json.load(f)
        logging.info(f"Loaded cluster lookup from {cluster_json_path}")
    else:
        logging.error(f"Cluster JSON file not found: {cluster_json_path}. "
                      "Ensure cluster_roots.py was run with matching --output_dir and --output_suffix.")
        raise FileNotFoundError(f"Cluster JSON file not found: {cluster_json_path}")

    logging.info(f"Parsing Voynich document: {voynich_doc_path}")
    document = Document(voynich_doc_path)
    cluster_lines = []

    for para_idx, para in enumerate(document.paragraphs):
        text = para.text.strip()
        if not text or not text.startswith('<'):
            continue

        parts = text.split()
        tag = parts[0]
        words = parts[1:]
        
        processed_line_words = [strip_suffix_local(word) for word in words]
        clusters = [cluster_lookup.get(word, None) for word in processed_line_words]
        cluster_lines.append((tag, clusters))
    logging.info(f"Processed {len(cluster_lines)} lines from the document.")

    # === Write output to CSV ===
    output_csv_base_filename = os.path.basename(args.output_csv_path_base) # ex: voynich_line_clusters.csv
    output_csv_path = generate_output_path(output_dir, output_csv_base_filename, output_file_suffix)

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Line Tag", "Cluster Sequence"])
        for tag, clusters in cluster_lines:
            writer.writerow([tag, clusters])
    logging.info(f"✅ Line cluster sequences saved to: {output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map Voynich manuscript lines to cluster IDs.")
    # Input data related arguments
    parser.add_argument("--voynich_doc_path", type=str, default='./data/AB.docx',
                        help="Path to the Voynich .docx transliteration.")
    # O cluster_json_path_base será combinado com output_dir e output_suffix para encontrar o arquivo correto
    parser.add_argument("--cluster_json_path_base", type=str, default="stripped_cluster_lookup.json",
                        help="Base filename for the stripped_word-to-cluster_id JSON file (expected in --output_dir).")

    # Processing related arguments
    parser.add_argument("--suffix_list", type=str, nargs='+',
                        default=['aiin', 'dy', 'in', 'chy', 'chey', 'edy', 'ey', 'y'],
                        help="List of suffixes (used if stripping is enabled).")
    parser.add_argument("--disable_suffix_stripping", action="store_true",
                        help="If set, assumes input cluster_lookup was generated without stripping and does not strip words itself.")

    # Output related arguments
    parser.add_argument("--output_dir", type=str, default='./results_custom', ### ALTERAÇÃO ###
                        help="Directory where input cluster_lookup is found and output CSV is saved.")
    parser.add_argument("--output_suffix", type=str, default="", ### ALTERAÇÃO ###
                        help="Suffix used for input cluster_lookup and appended to output CSV filename (e.g., '_no_stripping').")
    parser.add_argument("--output_csv_path_base", type=str, default="voynich_line_clusters.csv",
                        help="Base filename for the output CSV with line cluster sequences.")

    args = parser.parse_args()

    if args.output_suffix and args.output_suffix.startswith('_'):
        args.output_suffix = args.output_suffix[1:]
        
    main(args)