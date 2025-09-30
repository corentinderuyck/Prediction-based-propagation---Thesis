import json
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
import random

def read_jsonl_file_random_subset(file_path, p=None):
    """
    Lire un fichier JSONL très gros en streaming,
    et extraire un sous-ensemble aléatoire de taille N
    grâce au reservoir sampling.
    """
    if p is None:
        # Keep all objects if N is not specified or invalid
        objets = []
        decoder = json.JSONDecoder()

        with open(file_path, 'r') as file:
            for line in file:
                obj = decoder.decode(line.strip())
                if 'before' in obj and 'after' in obj:
                    objets.append(obj)

        return objets
    

    reservoir = []
    decoder = json.JSONDecoder()
    total_lines = 0
    sampled_lines = 0

    with open(file_path, 'r') as file:
        for line in file:
            obj = decoder.decode(line.strip())
            if 'before' in obj and 'after' in obj:
                total_lines += 1
                if random.random() < p:
                    reservoir.append(obj)
                    sampled_lines += 1

    print(f"Total graphs read: {total_lines}, sampled subset size: {sampled_lines}")
    return reservoir

def build_edge_list(graph_dict):
    """
    From {'0': [1,2], '1': [0,3]} build list of (u,v) edges.
    """
    edges = []
    for left_node, right_nodes in graph_dict.items():
        u = int(left_node)
        for v in right_nodes:
            edges.append((u, v))
    return edges

def create_tabular_features(before_dict, after_dict):
    """
    Create a tabular dataset from before/after dicts with node and edge features.
    Returns a pandas DataFrame where each row represents an edge with:
    - Left node features (normalized_degree, node_type, pagerank, betweenness, closeness, clustering, core)
    - Right node features (normalized_degree, node_type, pagerank, betweenness, closeness, clustering, core)
    - Edge features (Jaccard Coefficient, Adamic-Adar Index)
    - Label (1 if edge was removed, 0 otherwise)
    """
    edges_before = set(build_edge_list(before_dict))
    edges_after = set(build_edge_list(after_dict))

    # Get all the unique left and right nodes
    all_left_nodes = set(int(k) for k in before_dict.keys())
    all_right_nodes = set()
    for right_nodes in before_dict.values():
        all_right_nodes.update(right_nodes)
    
    # Mapping from nodes to indices
    left_to_idx = {node: i for i, node in enumerate(sorted(all_left_nodes))}
    right_to_idx = {node: i + len(all_left_nodes) for i, node in enumerate(sorted(all_right_nodes))}
    
    num_left = len(all_left_nodes)
    num_right = len(all_right_nodes)
    
    # Build NetworkX graph with integer-mapped nodes
    G = nx.Graph()
    for left_node, right_nodes in before_dict.items():
        left_idx = left_to_idx[int(left_node)]
        for right_node in right_nodes:
            right_idx = right_to_idx[right_node]
            G.add_edge(left_idx, right_idx)

    # If the graph is empty, there's nothing to process
    if not G.number_of_edges():
        return None

    # Compute node-level features
    pagerank_dict = nx.pagerank(G)
    betweenness_dict = nx.betweenness_centrality(G)
    closeness_dict = nx.closeness_centrality(G)
    clustering_dict = nx.clustering(G)
    core_dict = nx.core_number(G)
    
    # Compute degrees and normalize
    degree_dict = dict(G.degree())
    max_degree = max(degree_dict.values()) if degree_dict else 1
    normalized_degree_dict = {node: degree / max_degree for node, degree in degree_dict.items()}

    # Prepare edge data
    edge_data = []
    edges_uv = []
    
    for u, v in edges_before:
        u_idx = left_to_idx[u]
        v_idx = right_to_idx[v]
        edges_uv.append((u_idx, v_idx))
        
        # Edge label (1 if removed, 0 if kept)
        label = 1 if (u, v) not in edges_after else 0
        
        edge_data.append({
            'label': label
        })

    if not edge_data:
        return None

    # Compute edge-level features (Jaccard & Adamic-Adar)
    preds_jaccard = nx.jaccard_coefficient(G, edges_uv)
    preds_aa = nx.adamic_adar_index(G, edges_uv)

    jaccard_dict = {(u, v): p for u, v, p in preds_jaccard}
    aa_dict = {(u, v): p for u, v, p in preds_aa}

    # Create feature rows
    feature_rows = []
    
    for i, (edge_info, (u_idx, v_idx)) in enumerate(zip(edge_data, edges_uv)):
        # Left node features
        left_features = {
            'left_normalized_degree': normalized_degree_dict.get(u_idx, 0.0),
            'left_node_type': 0.0,  # Left nodes have type 0
            'left_pagerank': pagerank_dict.get(u_idx, 0.0),
            'left_betweenness': betweenness_dict.get(u_idx, 0.0),
            'left_closeness': closeness_dict.get(u_idx, 0.0),
            'left_clustering': clustering_dict.get(u_idx, 0.0),
            'left_core': core_dict.get(u_idx, 0.0)
        }
        
        # Right node features
        right_features = {
            'right_normalized_degree': normalized_degree_dict.get(v_idx, 0.0),
            'right_node_type': 1.0,  # Right nodes have type 1
            'right_pagerank': pagerank_dict.get(v_idx, 0.0),
            'right_betweenness': betweenness_dict.get(v_idx, 0.0),
            'right_closeness': closeness_dict.get(v_idx, 0.0),
            'right_clustering': clustering_dict.get(v_idx, 0.0),
            'right_core': core_dict.get(v_idx, 0.0)
        }
        
        # Edge features
        edge_features = {
            'jaccard_coefficient': jaccard_dict.get((u_idx, v_idx), 0.0),
            'adamic_adar_index': aa_dict.get((u_idx, v_idx), 0.0)
        }
        
        # Combine all features
        row = {**left_features, **right_features, **edge_features, 'label': edge_info['label']}
        feature_rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(feature_rows)
    
    return df

def process_all_graphs(objets):
    """
    Process all graphs and combine them into a single DataFrame.
    """
    all_dataframes = []
    
    pbar = tqdm(objets, desc="Processing graphs", leave=True, dynamic_ncols=True)
    
    for i, obj in enumerate(pbar):
        df = create_tabular_features(obj['before'], obj['after'])
        if df is not None:
            all_dataframes.append(df)
        
        pbar.set_postfix({"processed": len(all_dataframes)})
    
    if not all_dataframes:
        print("No valid graphs found.")
        return None
    
    # Combine all DataFrames
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    return combined_df

if __name__ == "__main__":
    # Data file path
    file_path = "../data/data_test.JSONL"
    
    p = 0.001  # Probability for random sampling
    
    # Lecture sous-échantillonnée aléatoire
    objets_subset = read_jsonl_file_random_subset(file_path, p=p)
    print(f"Using random subset of {len(objets_subset)} graphs for processing")
    
    # Process all graphs into tabular format
    feature_df = process_all_graphs(objets_subset)
    
    # Save to file if dataframe is not empty
    if feature_df is not None:
        feature_df.to_csv("tabular_features_test.csv", index=False)
        print("\nTabular feature extraction completed!")
    else:
        print("No features extracted.")
