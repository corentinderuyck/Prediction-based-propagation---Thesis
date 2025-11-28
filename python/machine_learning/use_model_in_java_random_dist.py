import json
import random
import socket
import logging
import os
import time
import threading
import bisect

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HistogramSampler:
    def __init__(self, json_path):
        logger.info(f"Loading histogram from {json_path}...")
        with open(json_path, "r") as f:
            hist = json.load(f)

        counts_dict = hist["counts"]

        items = sorted(
            counts_dict.items(),
            key=lambda kv: float(kv[0].split("-")[0])
        )

        self.bins = []
        self.cum_weights = []
        total = 0

        for interval_str, count in items:
            low_str, high_str = interval_str.split("-")
            low = float(low_str)
            high = float(high_str)
            total += count
            self.bins.append((low, high))
            self.cum_weights.append(total)

        self.total_weight = total
        logger.info(f"Histogram loaded with total weight {self.total_weight} "
                    f"and {len(self.bins)} bins.")

    def sample(self):
        r = random.uniform(0, self.total_weight)

        # Dichotomie
        idx = bisect.bisect_left(self.cum_weights, r)
        if idx >= len(self.bins):
            idx = len(self.bins) - 1

        low, high = self.bins[idx]

        return random.uniform(low, high)


class RandomPredictor:
    def __init__(self, threshold=0.5, sampler=None):
        self.threshold = threshold
        self.sampler = sampler
        logger.info(f"Random predictor initialized with threshold: {threshold}")
    
    def build_edge_list(self, graph_dict):
        """Build edge list from graph dictionary"""
        edges = []
        for left_node, right_nodes in graph_dict.items():
            u = int(left_node)
            for v in right_nodes:
                edges.append((u, v))
        return edges
    
    def predict(self, graph_json):
        """Prediction for each edge en utilisant un tirage pondéré par l'histogramme"""
        edges = self.build_edge_list(graph_json)
        if not edges:
            return {}
        
        edges_to_remove = {}
        for u, v in edges:
            random_value = self.sampler.sample()

            if random_value > self.threshold:
                edges_to_remove.setdefault(str(u), []).append(v)
        
        return edges_to_remove


# Configuration
THRESHOLD = 0.5
SOCKET_PATH = "/tmp/unix_socket_predictor"
HISTOGRAM_PATH = "../data/edge_probas_hist.json" 

# Time tracking
totalTime = 0.0

logger.info("Initializing histogram sampler...")
hist_sampler = HistogramSampler(HISTOGRAM_PATH)

# Initialize RandomPredictor
logger.info("Initializing random predictor...")
predictor = RandomPredictor(threshold=THRESHOLD, sampler=hist_sampler)
logger.info("Random predictor ready")


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
            logger.info("\nServer shutting down gracefully...")
        finally:
            if os.path.exists(SOCKET_PATH):
                os.remove(SOCKET_PATH)


if __name__ == "__main__":
    server()
