import json
import os
import sys
import random
from typing import Iterator, List, Optional

import pandas as pd
from tqdm import tqdm


def build_edge_list(graph_dict):
    edges = []
    for left_node, right_nodes in graph_dict.items():
        u = int(left_node)
        for v in right_nodes:
            edges.append((u, v))
    return edges


def create_edge_data(before_dict, after_dict):
    """Create edge list with labels for evaluation"""
    edges_before = set(build_edge_list(before_dict))
    edges_after = set(build_edge_list(after_dict))

    edge_list = []
    labels = []

    for u, v in sorted(edges_before):
        edge_list.append((u, v))
        # label = 1 if edge was removed, 0 if kept
        label = 1 if (u, v) not in edges_after else 0
        labels.append(label)

    return edge_list, labels


class StreamingJSONLDataset:
    """Dataset that reads JSONL file in streaming mode to save memory."""
    
    def __init__(self, jsonl_path: str):
        self.jsonl_path = jsonl_path
        
        # Count total lines for progress bar
        self.total_lines = 0
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for _ in f:
                self.total_lines += 1
    
    def __iter__(self):
        """Yield edge data for each graph."""
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
                edge_list, labels = create_edge_data(before, after)
                
                if edge_list:
                    yield edge_list, labels
            
            pbar.close()


class RandomPredictor:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
    
    def predict(self, edge_list):
        """Random prediction for each edge"""
        predictions = []
        for _ in edge_list:
            random_value = random.random()
            # Predict 1 (removed) if random_value > threshold
            pred = 1 if random_value > self.threshold else 0
            predictions.append(pred)
        return predictions


def evaluate_streaming(predictor, dataset, threshold):
    """
    Evaluate random predictor using streaming dataset to save memory.
    """
    tp_0 = fp_0 = tn_0 = fn_0 = 0
    tp_1 = fp_1 = tn_1 = fn_1 = 0

    for edge_list, labels in dataset:
        # Get random predictions
        predictions = predictor.predict(edge_list)
        
        for pred, target in zip(predictions, labels):
            pred_bool = (pred == 1)
            target_bool = (target == 1)
            
            # Class 1 (removed edges)
            if pred_bool and target_bool:
                tp_1 += 1
            elif pred_bool and not target_bool:
                fp_1 += 1
            elif not pred_bool and not target_bool:
                tn_1 += 1
            elif not pred_bool and target_bool:
                fn_1 += 1
            
            # Class 0 (non-removed edges)
            if not pred_bool and not target_bool:
                tp_0 += 1
            elif not pred_bool and target_bool:
                fp_0 += 1
            elif pred_bool and target_bool:
                tn_0 += 1
            elif pred_bool and not target_bool:
                fn_0 += 1

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


def run_evaluation_on_jsonl(jsonl_path, predictor):
    if not os.path.exists(jsonl_path):
        print(f"File not found: {jsonl_path}")
        return None
    
    # Check if file is empty
    file_size = os.path.getsize(jsonl_path)
    if file_size == 0:
        print(f"File is empty: {jsonl_path}")
        return None
    
    # Create streaming dataset
    dataset = StreamingJSONLDataset(jsonl_path)
    
    # Evaluate
    metrics = evaluate_streaming(predictor, dataset, predictor.threshold)
    return metrics


def main():
    INPUT_FOLDER = "../data/train_test_data"
    THRESHOLD = 0.5

    if not os.path.isdir(INPUT_FOLDER):
        print(f"Input folder not found: {INPUT_FOLDER}")
        sys.exit(1)

    jsonl_files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith((".jsonl", ".jl"))])
    if not jsonl_files:
        print(f"No JSONL files found in {INPUT_FOLDER}")
        sys.exit(1)

    # Initialize random predictor
    predictor = RandomPredictor(threshold=THRESHOLD)
    print(f"Random predictor initialized with threshold={THRESHOLD}")

    # summary rows to save at the end
    summary_rows = []

    # iterate files
    for fname in jsonl_files:
        path = os.path.join(INPUT_FOLDER, fname)
        print()
        print("=" * 80)
        print(f"Processing file: {path}")
        try:
            metrics = run_evaluation_on_jsonl(path, predictor)
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
        summary_path = os.path.join(INPUT_FOLDER, "evaluation_summary_random.csv")
        summary_df.to_csv(summary_path, index=False)
        print()
        print(f"Saved evaluation summary to: {summary_path}")
    else:
        print("No evaluations completed; no summary file created.")


if __name__ == '__main__':
    main()