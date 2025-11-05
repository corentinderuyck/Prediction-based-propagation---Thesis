import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_undirected, add_self_loops
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def read_jsonl_file(file_path):
    """
    Read a JSONL file and return a list of objects.
    """
    objets = []
    decoder = json.JSONDecoder()

    with open(file_path, 'r') as file:
        for line in file:
            obj = decoder.decode(line.strip())
            if 'before' in obj and 'after' in obj:
                objets.append(obj)

    return objets

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

def create_pyg_data(before_dict, after_dict):
    """
    Create a torch_geometric Data object from before/after dicts.
    Node features (x) are [normalized_degree, node_type].
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
    num_nodes = len(all_left_nodes) + len(all_right_nodes)
    
    # Build the edge index and labels
    edge_index_list = []
    edge_labels_list = []

    for u, v in sorted(edges_before):
        u_idx = left_to_idx[u]
        v_idx = right_to_idx[v]
        edge_index_list.append([u_idx, v_idx])
        label = 1 if (u, v) not in edges_after else 0
        edge_labels_list.append(label)

    if not edge_index_list:
        print("Warning: No edges found in before graph")
        return None

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_labels = torch.tensor(edge_labels_list, dtype=torch.float)

    edge_label_index = edge_index.clone()

    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    
    # Feature 1 : Normalized degree
    degrees = torch.zeros(num_nodes, dtype=torch.float)
    unique_nodes, counts = torch.unique(edge_index.flatten(), return_counts=True)
    degrees[unique_nodes] = counts.float()
    if degrees.max() > 0:
        degrees = degrees / degrees.max()
    
    # Feature 2 : Node type (0 = left, 1 = right)
    node_types = torch.zeros(num_nodes, dtype=torch.float)
    node_types[num_left:] = 1.0
    x = torch.cat([degrees.unsqueeze(1), node_types.unsqueeze(1)], dim=1)

    data = Data(x=x,
                edge_index=edge_index,
                edge_label=edge_labels,
                edge_label_index=edge_label_index
               )
    return data

class GraphSAGELinkPredictor(nn.Module):
    def __init__(self, hidden_channels, num_layers, dropout):
        super().__init__()
        
        self.input_dim = 2
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GraphSAGE layers
        self.sage_layers = nn.ModuleList()
        self.sage_layers.append(SAGEConv(self.input_dim, hidden_channels, project=True))

        for _ in range(num_layers - 1):
            self.sage_layers.append(SAGEConv(hidden_channels, hidden_channels, project=True))

        # Normalization layers
        self.layer_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.layer_norms.append(nn.LayerNorm(hidden_channels))
        
        # MLP for link prediction
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1)
        )
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Go through GraphSAGE layers
        for i, (sage_layer, layer_norm) in enumerate(zip(self.sage_layers, self.layer_norms)):
            x = sage_layer(x, edge_index)
            x = layer_norm(x)
            if i < len(self.sage_layers) - 1:
                # No dropout on the last layer
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Go through the MLP
        src, dst = data.edge_label_index
        edge_feats = torch.cat([x[src], x[dst]], dim=1)
        logits = self.edge_mlp(edge_feats).squeeze()
        
        return logits

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    
    # Progress bar
    pbar = tqdm(loader, desc="Training", leave=True, dynamic_ncols=True)
    
    for i, batch in enumerate(pbar, 1):
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, batch.edge_label)
        loss.backward()
        
        # Avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        avg_loss = total_loss / i
        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
    
    pbar.close()
    
    return total_loss / len(loader)


def evaluate(model, loader, threshold):
    model.eval()
    
    tp_0 = fp_0 = tn_0 = fn_0 = 0  # class 0
    tp_1 = fp_1 = tn_1 = fn_1 = 0  # class 1

    pbar = tqdm(loader, desc="Evaluating", leave=True, dynamic_ncols=True)

    with torch.no_grad():
        for batch in pbar:
            batch = batch
            
            # Forward pass
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

    pbar.close()

    total_samples = tp_1 + fp_1 + tn_1 + fn_1
    accuracy_global = (tp_1 + tn_1) / total_samples if total_samples > 0 else 0.0

    # Class 0 metrics
    precision_0 = tp_0 / (tp_0 + fp_0) if (tp_0 + fp_0) > 0 else 0.0
    recall_0 = tp_0 / (tp_0 + fn_0) if (tp_0 + fn_0) > 0 else 0.0
    f1_0 = (2 * precision_0 * recall_0) / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0.0
    support_0 = tp_0 + fn_0
    
    # Class 1 metrics
    precision_1 = tp_1 / (tp_1 + fp_1) if (tp_1 + fp_1) > 0 else 0.0
    recall_1 = tp_1 / (tp_1 + fn_1) if (tp_1 + fn_1) > 0 else 0.0
    f1_1 = (2 * precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0.0
    support_1 = tp_1 + fn_1
    
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


if __name__ == "__main__":
    # Data file path
    file_path = "../data/data_train_filtered.JSONL"

    # ======== Hyperparameters ========
    # GNN
    numberLayersGNN = 2
    hidden_channels = 4
    
    # MLP fixed parameters (3 layers ReLU + Dropout)
    dropout = 0.3

    # Other parameters
    test_size = 0.3
    epochs = 3
    batch_size = 32
    learning_rate = 1e-3
    weight_decay = 1e-5
    threshold = 0.5
    # ======== End Hyperparameters ========

    objets = read_jsonl_file(file_path)

    # Build the dataset
    all_data = []
    pbar = tqdm(objets, desc="Processing objects", leave=True, dynamic_ncols=True)
    for obj in pbar:
        data = create_pyg_data(obj['before'], obj['after'])
        if data is not None:
            all_data.append(data)
    pbar.close()

    if not all_data:
        print("No valid data found!")
        exit()

    print(f"Total graphs: {len(all_data)}")
    
    # Count the number of removed and total edges
    all_labels = []
    for data in all_data:
        all_labels.extend(data.edge_label.tolist())
    
    removed_edges = sum(all_labels)
    total_edges = len(all_labels)
    non_removed_edges = total_edges - removed_edges
    print(f"Removed edges: {removed_edges}/{total_edges} ({removed_edges/total_edges*100:.1f}%)")

    # Split train and test data
    train_data, test_data = train_test_split(all_data, test_size=test_size, random_state=1)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    
    # Initialize the model
    model = GraphSAGELinkPredictor(
        hidden_channels=hidden_channels,
        num_layers=numberLayersGNN,
        dropout=dropout
    )
    
    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Add weighted loss for imbalanced classes
    pos_weight = torch.tensor([non_removed_edges / removed_edges])

    # Avoid false positives for class 0 (non-removed edges)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Training loop
    print("Starting training...")
    
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion)

        # Evaluation
        train_metrics = evaluate(model, train_loader, threshold)
        test_metrics = evaluate(model, test_loader, threshold)

        print(f"Epoch {epoch:2d} - Loss: {train_loss:.4f}")
        print(f"  Train - Global Accuracy: {train_metrics['accuracy_global']:.4f}")
        print(f"  Test  - Global Accuracy: {test_metrics['accuracy_global']:.4f}")
        print()

        print(f"  Train - Class 0 (non-removed edges) - Precision: {train_metrics['class_0']['precision']:.4f}, Recall: {train_metrics['class_0']['recall']:.4f}, F1: {train_metrics['class_0']['f1']:.4f}, Support: {train_metrics['class_0']['support']}")
        print(f"  Test  - Class 0 (non-removed edges) - Precision: {test_metrics['class_0']['precision']:.4f}, Recall: {test_metrics['class_0']['recall']:.4f}, F1: {test_metrics['class_0']['f1']:.4f}, Support: {test_metrics['class_0']['support']}")
        print()

        print(f"  Train - Class 1 (removed edges) - Precision: {train_metrics['class_1']['precision']:.4f}, Recall: {train_metrics['class_1']['recall']:.4f}, F1: {train_metrics['class_1']['f1']:.4f}, Support: {train_metrics['class_1']['support']}")
        print(f"  Test  - Class 1 (removed edges) - Precision: {test_metrics['class_1']['precision']:.4f}, Recall: {test_metrics['class_1']['recall']:.4f}, F1: {test_metrics['class_1']['f1']:.4f}, Support: {test_metrics['class_1']['support']}")
        print()
        

    # Save the model
    torch.save(model.state_dict(), "model.pth")

    print("Training completed.")