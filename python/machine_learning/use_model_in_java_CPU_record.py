import json
import torch
import threading
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_undirected
import socket
import logging
import os
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROBA_JSON_PATH = "../data/edge_probas_hist.json"
BIN_WIDTH = 0.01          # 0.00-0.01, 0.01-0.02, ...
SAVE_EVERY = 50000


class ProbaHistogram:
    """
    In-memory histogram + JSON flush every SAVE_EVERY updates.
    Thread-safe.
    """
    def __init__(self, bin_width=0.01, save_every=500, path=PROBA_JSON_PATH):
        self.bin_width = float(bin_width)
        self.num_bins = int(round(1.0 / self.bin_width))
        self.counts = [0] * self.num_bins
        self.total_updates = 0
        self.save_every = int(save_every)
        self.path = path
        self.lock = threading.Lock()

        self._load_if_exists()

    def _load_if_exists(self):
        """Single read on startup if you want to resume an existing histogram."""
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r") as f:
                data = json.load(f)

            if abs(float(data.get("bin_width", self.bin_width)) - self.bin_width) > 1e-9:
                logger.warning("Different bin width found in existing JSON")
                return

            counts_dict = data.get("counts", {})
            new_counts = [0] * self.num_bins
            for i in range(self.num_bins):
                low = i * self.bin_width
                high = (i + 1) * self.bin_width
                key = f"{low:.2f}-{high:.2f}"
                if key in counts_dict:
                    new_counts[i] = int(counts_dict[key])

            self.counts = new_counts
            self.total_updates = int(data.get("total_updates", 0))
            logger.info(f"Histogram loaded from {self.path} (updates={self.total_updates}).")

        except Exception as e:
            logger.warning(f"Could not load existing histogram: {e}")

    def update(self, probs: torch.Tensor):
        """
        probs: 1D tensor of probabilities in [0,1]
        """
        if probs.numel() == 0:
            return

        probs_cpu = probs.detach().to("cpu")

        # Clamp so p=1.0 does not fall out of the last bin
        probs_cpu = probs_cpu.clamp(0.0, 1.0 - 1e-12)

        # Bin indices
        bin_idx = (probs_cpu / self.bin_width).floor().long()

        # Vectorized counting
        bc = torch.bincount(bin_idx, minlength=self.num_bins)

        with self.lock:
            for i in range(self.num_bins):
                self.counts[i] += int(bc[i])

            self.total_updates += 1
            if self.total_updates % self.save_every == 0:
                self._save_locked()

    def to_dict(self):
        d = {}
        for i, c in enumerate(self.counts):
            low = i * self.bin_width
            high = (i + 1) * self.bin_width
            d[f"{low:.2f}-{high:.2f}"] = c
        return d

    def _save_locked(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        payload = {
            "bin_width": self.bin_width,
            "counts": self.to_dict(),
            "total_updates": self.total_updates,
            "total_probs": int(sum(self.counts)),
            "timestamp": time.time(),
        }
        with open(self.path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Histogram saved")

    def flush(self):
        """Force an immediate save."""
        with self.lock:
            self._save_locked()


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


class OptimizedModelPredictor:
    def __init__(self, model_path, threshold=0.5,
                 bin_width=BIN_WIDTH, save_every=SAVE_EVERY, hist_path=PROBA_JSON_PATH):
        self.threshold = threshold
        
        self.device = torch.device('cpu')
        logger.info(f"Using device: {self.device}")

        self.model = GraphSAGELinkPredictor(
            hidden_channels=16,
            num_layers=2,
            dropout=0.3
        )

        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # New: in-memory histogram
        self.histogram = ProbaHistogram(
            bin_width=bin_width,
            save_every=save_every,
            path=hist_path
        )

    def build_edge_list(self, graph_dict):
        """Build edge list from graph dictionary."""
        edges = []
        for left_node, right_nodes in graph_dict.items():
            u = int(left_node)
            for v in right_nodes:
                edges.append((u, v))
        return edges

    def create_pyg_data(self, graph_dict):
        """Create PyG Data object from graph dictionary."""
        edges = sorted(set(self.build_edge_list(graph_dict)))

        all_left_nodes = set(int(k) for k in graph_dict.keys())
        all_right_nodes = set()
        for right_nodes in graph_dict.values():
            all_right_nodes.update(right_nodes)

        left_to_idx = {node: i for i, node in enumerate(sorted(all_left_nodes))}
        right_to_idx = {node: i + len(all_left_nodes) for i, node in enumerate(sorted(all_right_nodes))}

        num_left = len(all_left_nodes)
        
        edge_index = []
        original_edges = []

        for u, v in edges:
            u_idx = left_to_idx[u]
            v_idx = right_to_idx[v]
            edge_index.append([u_idx, v_idx])
            original_edges.append((u, v))

        if not edge_index:
            return None, None

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_label_index = edge_index.clone()
        num_nodes = num_left + len(all_right_nodes)
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)

        degrees = torch.zeros(num_nodes, dtype=torch.float)
        unique_nodes, counts = torch.unique(edge_index.flatten(), return_counts=True)
        degrees[unique_nodes] = counts.float()
        if degrees.max() > 0:
            degrees = degrees / degrees.max()

        node_types = torch.zeros(num_nodes, dtype=torch.float)
        node_types[num_left:] = 1.0

        x = torch.cat([degrees.unsqueeze(1), node_types.unsqueeze(1)], dim=1)

        data = Data(x=x, edge_index=edge_index, edge_label_index=edge_label_index)
        return data, original_edges

    def predict(self, graph_json):
        """CPU prediction for a graph + update in-memory histogram."""
        data, original_edges = self.create_pyg_data(graph_json)
        if data is None:
            return {}

        with torch.inference_mode():
            logits = self.model(data)
            probs = torch.sigmoid(logits)
            predictions = (probs > self.threshold).cpu().numpy()

        # New: update histogram instead of writing per-proba to disk
        self.histogram.update(probs)

        edges_to_remove = {}
        for i, (u, v) in enumerate(original_edges):
            if predictions[i]:
                edges_to_remove.setdefault(str(u), []).append(v)

        return edges_to_remove


# Configuration
MODEL_PATH = "../data/model.pth"
THRESHOLD = 0.5
SOCKET_PATH = "/tmp/unix_socket_predictor"

# Time tracking
totalTime = 0.0

# Initialize OptimizedModelPredictor
logger.info("Loading model...")
predictor = OptimizedModelPredictor(
    MODEL_PATH, 
    threshold=THRESHOLD,
    bin_width=BIN_WIDTH,
    save_every=SAVE_EVERY,
    hist_path=PROBA_JSON_PATH
)
logger.info("Model ready for inference (CPU only)")


def handle_client(conn):
    global totalTime
    buffer = b""
    try:
        while True:
            data = conn.recv(8192)
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

                # Special commands
                if "kill" in json_data:
                    logger.info("Received kill signal, flushing histogram then shutting down.")
                    predictor.histogram.flush()
                    os._exit(0)
                    return

                if "threshold" in json_data:
                    predictor.threshold = float(json_data["threshold"])
                    logger.info(f"Threshold updated to {predictor.threshold}")
                    continue

                if "time" in json_data:
                    response = json.dumps({"totalTime": totalTime * 1000}).encode("utf-8") + b"\n"
                    conn.sendall(response)
                    continue

                if "reset_time" in json_data:
                    totalTime = 0.0
                    logger.info("Python time reset to 0.0")
                    continue

                if "ping" in json_data:
                    logger.info("Received ping")
                    continue

                start_time = time.time()
                result = predictor.predict(json_data)
                end_time = time.time()
                totalTime += end_time - start_time

                response = json.dumps(result).encode("utf-8") + b"\n"
                conn.sendall(response)
                
    except Exception as e:
        logger.error(f"Exception in client handler: {e}", exc_info=True)
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
        except KeyboardInterrupt:
            logger.info("Server shutting down gracefully...")
        finally:
            # Flush histogram on normal shutdown too
            predictor.histogram.flush()
            if os.path.exists(SOCKET_PATH):
                os.remove(SOCKET_PATH)


if __name__ == "__main__":
    server()
