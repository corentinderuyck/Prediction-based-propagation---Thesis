import json
import torch
import threading
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINConv
from torch_geometric.utils import to_undirected, add_self_loops
import socket
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GraphGINLinkPredictor(nn.Module):
    def __init__(self, hidden_channels, num_layers, dropout, train_eps=True):
        super().__init__()
        self.input_dim = 2
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout

        self.gin_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        in_dim = self.input_dim
        for layer_idx in range(num_layers):
            # Petit MLP pour GIN : Lin -> ReLU -> Lin (sortie = hidden_channels)
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.gin_layers.append(GINConv(mlp, train_eps=train_eps))
            self.layer_norms.append(nn.LayerNorm(hidden_channels))
            in_dim = hidden_channels

        # MLP de prédiction de lien (inchangé)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i, (gin_layer, layer_norm) in enumerate(zip(self.gin_layers, self.layer_norms)):
            x = gin_layer(x, edge_index)
            x = layer_norm(x)
            if i < len(self.gin_layers) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        src, dst = data.edge_label_index

        edge_feats = torch.cat([x[src], x[dst]], dim=1)
        logits = self.edge_mlp(edge_feats).squeeze()
        return logits

class ModelPredictor:
    def __init__(self, model_path, threshold=0.5):
        self.threshold = threshold

        self.model = GraphGINLinkPredictor(
            hidden_channels=128,
            num_layers=3,
            dropout=0.3,
            train_eps=True
        )

        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def build_edge_list(self, graph_dict):
        edges = []
        for left_node, right_nodes in graph_dict.items():
            u = int(left_node)
            for v in right_nodes:
                edges.append((u, v))
        return edges

    def create_pyg_data(self, graph_dict):
        edges = set(self.build_edge_list(graph_dict))

        all_left_nodes = set(int(k) for k in graph_dict.keys())
        all_right_nodes = set()
        for right_nodes in graph_dict.values():
            all_right_nodes.update(right_nodes)

        left_to_idx = {node: i for i, node in enumerate(sorted(all_left_nodes))}
        right_to_idx = {node: i + len(all_left_nodes) for i, node in enumerate(sorted(all_right_nodes))}

        num_left = len(all_left_nodes)
        num_nodes = num_left + len(all_right_nodes)

        edges_sorted = sorted(edges)

        edge_index_list = []
        original_edges = []
        for u, v in edges_sorted:
            u_idx = left_to_idx[u]
            v_idx = right_to_idx[v]
            edge_index_list.append([u_idx, v_idx])
            original_edges.append((u, v))

        if not edge_index_list:
            return None, None

        edge_index_orig = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_label_index = edge_index_orig.clone()

        edge_index_for_mp = to_undirected(edge_index_orig, num_nodes=num_nodes)

        # Feature 1: normalized degree (computed on propagation graph)
        degrees = torch.zeros(num_nodes, dtype=torch.float)
        unique_nodes, counts = torch.unique(edge_index_for_mp.flatten(), return_counts=True)
        degrees[unique_nodes] = counts.float()
        if degrees.max() > 0:
            degrees = degrees / degrees.max()

        # Feature 2: node type
        node_types = torch.zeros(num_nodes, dtype=torch.float)
        node_types[num_left:] = 1.0

        x = torch.cat([degrees.unsqueeze(1), node_types.unsqueeze(1)], dim=1)

        data = Data(
            x=x,
            edge_index=edge_index_for_mp, 
            edge_label_index=edge_label_index
        )
        return data, original_edges
    
    def predict(self, graph_json):
        data, original_edges = self.create_pyg_data(graph_json)
        if data is None:
            return {}

        device = next(self.model.parameters()).device
        data = data.to(device)

        with torch.no_grad():
            logits = self.model(data)
            probs = torch.sigmoid(logits)
            preds = (probs > self.threshold).cpu().numpy()

        edges_to_remove = {}
        for i, (u, v) in enumerate(original_edges):
            if preds[i]:
                edges_to_remove.setdefault(str(u), []).append(v)

        return edges_to_remove


# Configuration
MODEL_PATH = "../data/model.pth"
THRESHOLD = 0.5
SOCKET_PATH = "/tmp/unix_socket_predictor"

# Init ModelPredictor
logger.info("Loading model...")
predictor = ModelPredictor(MODEL_PATH, threshold=THRESHOLD)

# Unix Domain Socket server
def handle_client(conn):
    global stop_server
    buffer = b""
    try:
        while True:
            data = conn.recv(256)
            if not data:
                break
            buffer += data
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                try:
                    json_data = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
                    response = json.dumps({"error": "Invalid JSON"}).encode("utf-8") + b"\n"
                    conn.sendall(response)
                    continue

                if "kill" in json_data:
                    logger.info("Received kill signal, shutting down server.")
                    os._exit(0)
                    return

                if "threshold" in json_data:
                    predictor.threshold = float(json_data["threshold"])
                    logger.info(f"Threshold updated to {predictor.threshold}")
                    continue


                result = predictor.predict(json_data)
                response = json.dumps(result).encode("utf-8") + b"\n"
                conn.sendall(response)
    except Exception as e:
        logger.error(f"Exception in client handler: {e}")
    finally:
        conn.close()
        logger.info("Client disconnected")

def server():
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server_sock:
        server_sock.bind(SOCKET_PATH)
        server_sock.listen()
        logger.info(f"Unix Domain Socket server listening on {SOCKET_PATH}")

        try:
            while True:
                conn, _ = server_sock.accept()
                logger.info("Client connected")
                threading.Thread(target=handle_client, args=(conn,), daemon=True).start()
        finally:
            if os.path.exists(SOCKET_PATH):
                os.remove(SOCKET_PATH)

if __name__ == "__main__":
    server()