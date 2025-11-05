import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import networkx as nx

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
    Create a torch_geometric Data object from before/after dicts with node and edge features.
    Node Features:
        - normalized_degree, node_type, pagerank, betweenness centrality, closeness centrality, clustering coefficient, core number
    Edge Features:
        - Jaccard Coefficient, Adamic-Adar Index
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
    num_nodes = num_left + num_right
    
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

    # Edge index & labels
    edge_index = []
    edge_labels = []
    edges_uv = [] # Store original (u,v) nodes for feature calculation
    for u, v in edges_before:
        u_idx = left_to_idx[u]
        v_idx = right_to_idx[v]
        edge_index.append([u_idx, v_idx])
        edge_labels.append(1 if (u, v) not in edges_after else 0)
        edges_uv.append((u_idx, v_idx))

    if not edge_index:
        return None

    # Compute edge-level features (Jaccard & Adamic-Adar)
    preds_jaccard = nx.jaccard_coefficient(G, edges_uv)
    preds_aa = nx.adamic_adar_index(G, edges_uv)

    jaccard_dict = {(u, v): p for u, v, p in preds_jaccard}
    aa_dict = {(u, v): p for u, v, p in preds_aa}

    edge_attr = []
    for u, v in edges_uv:
        jacc = jaccard_dict.get((u, v), 0.0)
        aa = aa_dict.get((u, v), 0.0)
        edge_attr.append([jacc, aa])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_labels = torch.tensor(edge_labels, dtype=torch.float)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Degrees (node features)
    degrees = torch.zeros(num_nodes, dtype=torch.float)
    for i in range(num_nodes):
        degrees[i] = ((edge_index[0] == i) | (edge_index[1] == i)).sum().float()
    degrees = degrees / degrees.max() if degrees.max() > 0 else degrees
    degrees = degrees.unsqueeze(1)

    # Node types (node features)
    node_types = torch.zeros(num_nodes, dtype=torch.float)
    node_types[num_left:] = 1.0
    node_types = node_types.unsqueeze(1)

    # Other node features
    pagerank_values = torch.tensor([pagerank_dict.get(i, 0) for i in range(num_nodes)], dtype=torch.float).unsqueeze(1)
    betweenness_values = torch.tensor([betweenness_dict.get(i, 0) for i in range(num_nodes)], dtype=torch.float).unsqueeze(1)
    closeness_values = torch.tensor([closeness_dict.get(i, 0) for i in range(num_nodes)], dtype=torch.float).unsqueeze(1)
    clustering_values = torch.tensor([clustering_dict.get(i, 0) for i in range(num_nodes)], dtype=torch.float).unsqueeze(1)
    core_values = torch.tensor([core_dict.get(i, 0) for i in range(num_nodes)], dtype=torch.float).unsqueeze(1)

    # Combine all node features
    x = torch.cat([
        degrees,
        node_types,
        pagerank_values,
        betweenness_values,
        closeness_values,
        clustering_values,
        core_values
    ], dim=1)

    data = Data(x=x, edge_index=edge_index, edge_label=edge_labels, edge_attr=edge_attr)
    return data

class GraphSAGELinkPredictor(nn.Module):
    def __init__(self, input_dim, hidden_channels, num_layers, dropout, edge_features_dim):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.edge_features_dim = edge_features_dim
        
        # GraphSAGE layers
        self.sage_layers = nn.ModuleList()
        self.sage_layers.append(SAGEConv(self.input_dim, hidden_channels, project=True))

        for _ in range(num_layers - 1):
            self.sage_layers.append(SAGEConv(hidden_channels, hidden_channels, project=True))

        self.layer_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.layer_norms.append(nn.LayerNorm(hidden_channels))
        
        # Adjust MLP input dimension to include edge features
        self.mlp_input_dim = 2 * hidden_channels + edge_features_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.mlp_input_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1)
        )
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        for i, (sage_layer, layer_norm) in enumerate(zip(self.sage_layers, self.layer_norms)):
            x = sage_layer(x, edge_index)
            x = layer_norm(x)
            if i < len(self.sage_layers) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        src, dst = data.edge_index
        
        # Concatenate node embeddings and edge features
        edge_feats = torch.cat([x[src], x[dst], edge_attr], dim=1)
        logits = self.edge_mlp(edge_feats).squeeze()
        
        return logits

class AntiFalsePositiveLoss(nn.Module):
    def __init__(self, fp_penalty, pos_weight):
        super(AntiFalsePositiveLoss, self).__init__()
        self.fp_penalty = fp_penalty
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        base_loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none'
        )

        probs = torch.sigmoid(logits)

        # Penalise false positives
        fp_penalty_term = self.fp_penalty * probs * (1 - targets)

        # Total loss
        total_loss = base_loss + fp_penalty_term

        return total_loss.mean()

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
    
    return total_loss / len(loader)

def evaluate(model, loader, threshold, mode):
    model.eval()
    
    tp_0 = fp_0 = tn_0 = fn_0 = 0  # class 0
    tp_1 = fp_1 = tn_1 = fn_1 = 0  # class 1

    pbar = tqdm(loader, desc="Evaluating on " + mode + " set", leave=True, dynamic_ncols=True)
    
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
    numberLayersGNN = 4
    hidden_channels = 128
    
    # MLP fixed parameters (3 layers ReLU + Dropout)
    dropout = 0.3

    # Loss function parameters
    fp_penalty = 0

    # Other parameters
    test_size = 0.3
    epochs = 10
    batch_size = 32
    learning_rate = 1e-3
    weight_decay = 1e-5
    threshold = 0.5
    # ======== End Hyperparameters ========

    objets = read_jsonl_file(file_path)

    pbar = tqdm(objets, desc="Loading data", leave=True, dynamic_ncols=True)

    # Build the dataset
    all_data = []
    for obj in pbar:
        data = create_pyg_data(obj['before'], obj['after'])
        if data is not None:
            all_data.append(data)

        pbar.set_postfix({"loaded": len(all_data)})

    if not all_data:
        print("No valid graphs found. Exiting.")
        exit()

    input_dim = all_data[0].x.shape[1]
    edge_features_dim = all_data[0].edge_attr.shape[1]

    print(f"Total number of graphs: {len(all_data)}")
    
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
    
    # Initialize the model with the new parameter
    model = GraphSAGELinkPredictor(
        input_dim=input_dim,
        hidden_channels=hidden_channels,
        num_layers=numberLayersGNN,
        dropout=dropout,
        edge_features_dim=edge_features_dim
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Add weighted loss for imbalanced classes
    if removed_edges > 0:
        pos_weight = torch.tensor([non_removed_edges / removed_edges])
    else:
        pos_weight = torch.tensor([1.0])

    # Loss function
    criterion = AntiFalsePositiveLoss(fp_penalty, pos_weight)

    # Training loop
    for epoch in range(1, epochs + 1):
        print(f"Training epoch {epoch}/{epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion)

        # Evaluation
        print(f"Evaluating epoch {epoch}/{epochs}")
        train_metrics = evaluate(model, train_loader, threshold, "train")
        test_metrics = evaluate(model, test_loader, threshold, "test")

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

    print("Training completed")