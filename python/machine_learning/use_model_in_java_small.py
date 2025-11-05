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
    def __init__(self, model_path, threshold=0.5, device='auto', use_amp=True):
        self.threshold = threshold
        self.use_amp = use_amp
        
        # Automatic device detection
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            
            # GPU optimizations
            torch.backends.cudnn.benchmark = True  # Auto-tune for fixed input sizes
            torch.backends.cuda.matmul.allow_tf32 = True  # TensorFloat-32 for matmul
            torch.backends.cudnn.allow_tf32 = True  # TensorFloat-32 for convolutions
            
            # High performance mixed precision
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high')
            
            logger.info(f"GPU optimizations enabled (cuDNN benchmark, TF32, AMP={use_amp})")
        
        # Load model
        self.model = GraphSAGELinkPredictor(
            hidden_channels=4,
            num_layers=2,
            dropout=0.3
        )

        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def build_edge_list(self, graph_dict):
        """Build edge list from graph dictionary"""
        edges = []
        for left_node, right_nodes in graph_dict.items():
            u = int(left_node)
            for v in right_nodes:
                edges.append((u, v))
        return edges

    def create_pyg_data(self, graph_dict):
        """Create PyG Data object from graph dictionary"""
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

        # Feature 1: Normalized degree computed on undirected graph
        degrees = torch.zeros(num_nodes, dtype=torch.float)
        unique_nodes, counts = torch.unique(edge_index.flatten(), return_counts=True)
        degrees[unique_nodes] = counts.float()
        if degrees.max() > 0:
            degrees = degrees / degrees.max()

        # Feature 2: Node type (left or right)
        node_types = torch.zeros(num_nodes, dtype=torch.float)
        node_types[num_left:] = 1.0

        x = torch.cat([degrees.unsqueeze(1), node_types.unsqueeze(1)], dim=1)

        data = Data(x=x, edge_index=edge_index, edge_label_index=edge_label_index)
        return data, original_edges

    def predict(self, graph_json):
        """Optimized prediction for a single graph"""
        data, original_edges = self.create_pyg_data(graph_json)
        if data is None:
            return {}

        # Asynchronous transfer to GPU (non_blocking=True)
        data = data.to(self.device, non_blocking=True)
        
        # Inference with or without mixed precision
        with torch.inference_mode():
            if self.use_amp and self.device.type == 'cuda':
                # Automatic mixed precision (FP16/FP32 mix)
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    logits = self.model(data)
            else:
                logits = self.model(data)
            
            # Post-processing on GPU then minimal transfer to CPU
            probs = torch.sigmoid(logits)
            predictions = (probs > self.threshold)
            
            # Transfer only boolean result (small)
            predictions = predictions.cpu().numpy()

        # Reconstruct result dictionary
        edges_to_remove = {}
        for i, (u, v) in enumerate(original_edges):
            if predictions[i]:
                edges_to_remove.setdefault(str(u), []).append(v)

        return edges_to_remove


# Configuration
MODEL_PATH = "../data/model.pth"
THRESHOLD = 0.5
SOCKET_PATH = "/tmp/unix_socket_predictor"
DEVICE = 'cpu'

# GPU optimizations
USE_AMP = True

# Time tracking
totalTime = 0.0

# Initialize OptimizedModelPredictor
logger.info("Loading model...")
predictor = OptimizedModelPredictor(
    MODEL_PATH, 
    threshold=THRESHOLD, 
    device=DEVICE,
    use_amp=USE_AMP
)
logger.info("Model ready for inference")

# Unix Domain Socket server
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
                    logger.info("Received kill signal, shutting down server.")
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
                
                # GPU synchronization for precise time measurement
                if predictor.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start_time = time.time()
                result = predictor.predict(json_data)
                
                # Synchronization for precise measurement
                if predictor.device.type == 'cuda':
                    torch.cuda.synchronize()
                
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
            logger.info("\nServer shutting down gracefully...")
        finally:
            if os.path.exists(SOCKET_PATH):
                os.remove(SOCKET_PATH)

if __name__ == "__main__":
    server()