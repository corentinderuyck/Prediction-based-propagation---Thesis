import json
import os
import sys
from typing import Iterator, List, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_undirected
from tqdm import tqdm


def build_edge_list(graph_dict):
    edges = []
    for left_node, right_nodes in graph_dict.items():
        u = int(left_node)
        for v in right_nodes:
            edges.append((u, v))
    return edges


def create_pyg_data(before_dict, after_dict):
    edges_before = set(build_edge_list(before_dict))
    edges_after = set(build_edge_list(after_dict))

    all_left_nodes = set(int(k) for k in before_dict.keys())
    all_right_nodes = set()
    for right_nodes in before_dict.values():
        all_right_nodes.update(right_nodes)

    left_to_idx = {node: i for i, node in enumerate(sorted(all_left_nodes))}
    right_to_idx = {node: i + len(all_left_nodes) for i, node in enumerate(sorted(all_right_nodes))}

    num_left = len(all_left_nodes)
    num_nodes = len(all_left_nodes) + len(all_right_nodes)

    edge_index_list = []
    edge_labels_list = []
    orig_edge_list = []

    for u, v in sorted(edges_before):
        if u not in left_to_idx or v not in right_to_idx:
            continue
        u_idx = left_to_idx[u]
        v_idx = right_to_idx[v]
        edge_index_list.append([u_idx, v_idx])
        label = 1 if (u, v) not in edges_after else 0
        edge_labels_list.append(label)
        orig_edge_list.append((u, v))

    if not edge_index_list:
        return None

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_labels = torch.tensor(edge_labels_list, dtype=torch.float)
    edge_label_index = edge_index.clone()

    edge_index = to_undirected(edge_index, num_nodes=num_nodes)

    degrees = torch.zeros(num_nodes, dtype=torch.float)
    unique_nodes, counts = torch.unique(edge_index.flatten(), return_counts=True)
    degrees[unique_nodes] = counts.float()
    if degrees.max() > 0:
        degrees = degrees / degrees.max()

    node_types = torch.zeros(num_nodes, dtype=torch.float)
    node_types[num_left:] = 1.0

    x = torch.cat([degrees.unsqueeze(1), node_types.unsqueeze(1)], dim=1)

    data = Data(x=x,
                edge_index=edge_index,
                edge_label=edge_labels,
                edge_label_index=edge_label_index)

    data.orig_edge_list = orig_edge_list
    return data


class StreamingJSONLDataset:
    """Dataset that reads JSONL file in streaming mode to save memory."""
    
    def __init__(self, jsonl_path: str, batch_size: int = 32):
        self.jsonl_path = jsonl_path
        self.batch_size = batch_size
        
        # Count total lines for progress bar
        self.total_lines = 0
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for _ in f:
                self.total_lines += 1
    
    def __iter__(self) -> Iterator[Batch]:
        """Yield batches of Data objects directly from file."""
        batch_buffer = []
        line_num = 0
        
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            pbar = tqdm(total=self.total_lines, desc=f"Processing {os.path.basename(self.jsonl_path)}", 
                       leave=True, dynamic_ncols=True)
            
            for line in f:
                line_num += 1
                pbar.update(1)
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    obj = json.loads(line)
                except Exception as e:
                    print(f"Line {line_num}: JSON parse error: {e} - skipped")
                    continue
                
                if 'before' not in obj or 'after' not in obj:
                    print(f"Line {line_num}: missing 'before'/'after' keys - skipped")
                    continue
                
                before = obj['before'] or {}
                after = obj['after'] or {}
                data = create_pyg_data(before, after)
                
                if data is not None:
                    data.graph_index = line_num
                    batch_buffer.append(data)
                    
                    # Yield batch when buffer is full
                    if len(batch_buffer) >= self.batch_size:
                        batch = Batch.from_data_list(batch_buffer)
                        yield batch
                        batch_buffer = []
            
            # Yield remaining data
            if batch_buffer:
                batch = Batch.from_data_list(batch_buffer)
                yield batch
            
            pbar.close()


class GraphSAGELinkPredictor(nn.Module):
    def __init__(self, hidden_channels, num_layers, dropout):
        super().__init__()
        self.input_dim = 2
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout

        self.sage_layers = nn.ModuleList()
        self.sage_layers.append(SAGEConv(self.input_dim, hidden_channels, project=True))
        for _ in range(num_layers - 1):
            self.sage_layers.append(SAGEConv(hidden_channels, hidden_channels, project=True))

        self.layer_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.layer_norms.append(nn.LayerNorm(hidden_channels))

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i, (sage_layer, layer_norm) in enumerate(zip(self.sage_layers, self.layer_norms)):
            x = sage_layer(x, edge_index)
            x = layer_norm(x)
            if i < len(self.sage_layers) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        src, dst = data.edge_label_index
        edge_feats = torch.cat([x[src], x[dst]], dim=1)
        logits = self.edge_mlp(edge_feats).squeeze()
        return logits


def evaluate_streaming(model, dataset, threshold, device):
    """
    Evaluate model using streaming dataset to save memory.
    """
    model.eval()

    tp_0 = fp_0 = tn_0 = fn_0 = 0
    tp_1 = fp_1 = tn_1 = fn_1 = 0

    with torch.no_grad():
        for batch in dataset:
            # move to device
            try:
                batch = batch.to(device)
            except Exception:
                pass

            logits = model(batch)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold)

            targets = batch.edge_label.bool()

            # Class 1 (removed edges)
            tp_1 += (preds & targets).sum().item()
            fp_1 += (preds & ~targets).sum().item()
            tn_1 += (~preds & ~targets).sum().item()
            fn_1 += (~preds & targets).sum().item()

            # Class 0 (non-removed edges)
            tp_0 += (~preds & ~targets).sum().item()
            fp_0 += (~preds & targets).sum().item()
            tn_0 += (preds & targets).sum().item()
            fn_0 += (preds & ~targets).sum().item()

    total_samples = tp_1 + fp_1 + tn_1 + fn_1
    accuracy_global = (tp_1 + tn_1) / total_samples if total_samples > 0 else 0.0

    # Class 0 metrics
    precision_0 = tp_0 / (tp_0 + fp_0) if (tp_0 + fp_0) > 0 else 0.0
    recall_0 = tp_0 / (tp_0 + fn_0) if (tp_0 + fn_0) > 0 else 0.0
    f1_0 = (2 * precision_0 * recall_0) / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0.0
    support_0 = int(tp_0 + fn_0)

    # Class 1 metrics
    precision_1 = tp_1 / (tp_1 + fp_1) if (tp_1 + fp_1) > 0 else 0.0
    recall_1 = tp_1 / (tp_1 + fn_1) if (tp_1 + fn_1) > 0 else 0.0
    f1_1 = (2 * precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0.0
    support_1 = int(tp_1 + fn_1)

    return {
        'accuracy_global': accuracy_global,
        'class_0': {
            'precision': precision_0,
            'recall': recall_0,
            'f1': f1_0,
            'support': support_0
        },
        'class_1': {
            'precision': precision_1,
            'recall': recall_1,
            'f1': f1_1,
            'support': support_1
        }
    }


def print_metrics(metrics):
    print()
    print(f"Global Accuracy: {metrics['accuracy_global']:.4f}")
    print(f"Class 0 (non-removed edges) - Precision: {metrics['class_0']['precision']:.4f}, Recall: {metrics['class_0']['recall']:.4f}, F1: {metrics['class_0']['f1']:.4f}, Support: {metrics['class_0']['support']}")
    print(f"Class 1 (removed edges)     - Precision: {metrics['class_1']['precision']:.4f}, Recall: {metrics['class_1']['recall']:.4f}, F1: {metrics['class_1']['f1']:.4f}, Support: {metrics['class_1']['support']}")


def run_evaluation_on_jsonl(jsonl_path, model, device, batch_size, threshold):
    if not os.path.exists(jsonl_path):
        print(f"File not found: {jsonl_path}")
        return None
    
    # Check if file is empty
    file_size = os.path.getsize(jsonl_path)
    if file_size == 0:
        print(f"File is empty: {jsonl_path}")
        return None
    
    # Create streaming dataset
    dataset = StreamingJSONLDataset(jsonl_path, batch_size=batch_size)
    
    # Evaluate
    metrics = evaluate_streaming(model, dataset, threshold, device)
    return metrics


def main():
    INPUT_FOLDER = "../data/test_data"
    MODEL_PATH = "../data/model.pth"
    BATCH_SIZE = 32
    THRESHOLD = 0.5
    HIDDEN_CHANNELS = 16
    NUM_LAYERS = 2
    DROPOUT = 0.3
    DEVICE = 'cpu'

    if not os.path.isdir(INPUT_FOLDER):
        print(f"Input folder not found: {INPUT_FOLDER}")
        sys.exit(1)

    jsonl_files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith((".jsonl", ".jl"))])
    if not jsonl_files:
        print(f"No JSONL files found in {INPUT_FOLDER}")
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        sys.exit(1)

    # load model once
    model = GraphSAGELinkPredictor(hidden_channels=HIDDEN_CHANNELS, num_layers=NUM_LAYERS, dropout=DROPOUT)
    model.to(DEVICE)
    print(f"Loading model weights from {MODEL_PATH}...")
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    try:
        model.load_state_dict(state)
    except Exception:
        # try adapt keys (module.)
        try:
            model.load_state_dict({k.replace('module.', ''): v for k, v in state.items()}, strict=False)
        except Exception as e:
            print("Failed to load model weights:", e)
            sys.exit(1)

    # summary rows to save at the end
    summary_rows = []

    # iterate files
    for fname in jsonl_files:
        path = os.path.join(INPUT_FOLDER, fname)
        print()
        print("=" * 80)
        print(f"Processing file: {path}")
        try:
            metrics = run_evaluation_on_jsonl(path, model, DEVICE, BATCH_SIZE, THRESHOLD)
            if metrics is None:
                continue
            print_metrics(metrics)

            # append to summary
            summary_rows.append({
                'file': fname,
                'accuracy_global': metrics['accuracy_global'],
                'precision_0': metrics['class_0']['precision'],
                'recall_0': metrics['class_0']['recall'],
                'f1_0': metrics['class_0']['f1'],
                'support_0': metrics['class_0']['support'],
                'precision_1': metrics['class_1']['precision'],
                'recall_1': metrics['class_1']['recall'],
                'f1_1': metrics['class_1']['f1'],
                'support_1': metrics['class_1']['support']
            })

        except Exception as e:
            print(f"Error processing {fname}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # save summary CSV if we have results
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(INPUT_FOLDER, "evaluation_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print()
        print(f"Saved evaluation summary to: {summary_path}")
    else:
        print("No evaluations completed; no summary file created.")


if __name__ == '__main__':
    main()