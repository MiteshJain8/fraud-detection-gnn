from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import torch
import json
import numpy as np
from torch_geometric.utils import degree

# Import necessary functions and classes
try:
    from .data_preprocess import preprocess_beneficiary, preprocess_claims
except ImportError:
    print("Warning: data_preprocess.py not found or contains errors.")
    preprocess_beneficiary = preprocess_claims = None

try:
    from .graph_builder import build_graph
except ImportError:
    print("Warning: graph_builder.py not found or contains errors.")
    build_graph = None

try:
    from .train import train_model
except ImportError:
    print("Warning: train.py not found or contains errors.")
    train_model = None

try:
    from .model import GCNEncoder, FraudGAE
except ImportError:
    print("Warning: model.py not found or contains errors.")
    GCNEncoder = FraudGAE = None

from .config import (
    BENEF_2008_CSV, BENEF_2010_CSV, BENEF_2008_PROC, BENEF_2010_PROC,
    CLAIMS_PROC, GRAPH_PT, NODE_MAP_JSON, MODEL_SAVE_PATH
)

# --- Model Loading and Anomaly Scoring Implementation ---

def load_model_and_graph(embedding_dim=32): # embedding_dim should match training
    """Loads the trained GAE model and graph data."""
    if not GCNEncoder or not FraudGAE:
        print("Error: Model classes not imported.")
        return None, None, None

    # Load graph data
    try:
        data = torch.load(GRAPH_PT)
        print(f"Loaded graph data from {GRAPH_PT}")
    except FileNotFoundError:
        print(f"Error: Graph data file not found at {GRAPH_PT}.")
        return None, None, None

    # Load node mapping
    try:
        with open(NODE_MAP_JSON, 'r') as f:
            node_mapping = json.load(f)
        # Create reverse mapping (int_id -> original_id)
        reverse_node_mapping = {v: k for k, v in node_mapping.items()}
        print(f"Loaded node mapping from {NODE_MAP_JSON}")
    except FileNotFoundError:
        print(f"Error: Node mapping file not found at {NODE_MAP_JSON}.")
        return None, None, None

    # Initialize model structure
    in_channels = data.num_node_features
    encoder = GCNEncoder(in_channels, embedding_dim)
    model = FraudGAE(encoder)

    # Load trained state dict
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        model.eval() # Set model to evaluation mode
        print(f"Loaded trained model state from {MODEL_SAVE_PATH}")
        return model, data, reverse_node_mapping
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_SAVE_PATH}. Train the model first.")
        return None, None, None
    except Exception as e:
        print(f"Error loading model state: {e}")
        return None, None, None

def score_anomalies(model, data, reverse_node_mapping, top_n=20):
    """Calculates anomaly scores for nodes based on reconstruction error."""
    if not model or not data or not reverse_node_mapping:
        print("Error: Model, data, or mapping not provided for scoring.")
        return []

    print("Calculating anomaly scores...")
    with torch.no_grad(): # Disable gradient calculation for inference
        z = model.encode(data.x, data.edge_index)

        # Calculate reconstruction error per node
        # One way: Calculate the average reconstruction error (BCE loss) for edges connected to each node
        # This requires iterating through nodes or using scatter operations

        # Simpler approach: Anomaly score based on reconstruction error of node features (if decoder exists)
        # Since GAE uses inner product decoder, let's calculate node-level anomaly score
        # based on the average reconstruction error of its incident edges.

        recon_error_per_edge = []
        pos_edge_index = data.edge_index
        num_edges = pos_edge_index.size(1)

        # Calculate reconstruction probability for positive edges
        pos_logits = model.decode(z, pos_edge_index)
        pos_loss = -F.logsigmoid(pos_logits).mean(dim=0) # Average loss per edge

        # We need to associate this loss back to individual nodes.
        # Calculate average loss for edges connected to each node.
        # Use degree to normalize
        node_degrees = degree(pos_edge_index[0], num_nodes=data.num_nodes) + degree(pos_edge_index[1], num_nodes=data.num_nodes)
        node_recon_error = torch.zeros(data.num_nodes)

        # Scatter add the loss to connected nodes
        # Note: This is an approximation. A more rigorous approach might be needed.
        # Calculate loss per edge (using BCE with logits directly)
        pos_edge_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits), reduction='none')

        # Sum loss for edges connected to each node
        node_recon_error.scatter_add_(0, pos_edge_index[0], pos_edge_loss)
        node_recon_error.scatter_add_(0, pos_edge_index[1], pos_edge_loss)

        # Average the error by degree (avoid division by zero for isolated nodes)
        node_recon_error = node_recon_error / (node_degrees + 1e-6)

        # Get top N anomalies
        scores = node_recon_error.cpu().numpy()
        sorted_indices = np.argsort(scores)[::-1] # Sort descending

        anomalies = []
        for i in range(min(top_n, data.num_nodes)):
            node_idx = sorted_indices[i]
            original_id = reverse_node_mapping.get(node_idx, f"Unknown_Node_{node_idx}")
            score = scores[node_idx]
            anomalies.append({'id': int(node_idx), 'node': original_id, 'score': float(score)})

    print(f"Identified top {len(anomalies)} anomalies.")
    return anomalies

# --- FastAPI App Definition ---
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints ---

@app.post('/preprocess')
def preprocess_endpoint(): # Renamed to avoid conflict
    if preprocess_beneficiary and preprocess_claims:
        if not os.path.exists(BENEF_2008_CSV) or not os.path.exists(BENEF_2010_CSV) or not os.path.exists(CLAIMS_CSV):
             return {'status': 'error', 'message': 'Raw data files not found.'}
        try:
            preprocess_beneficiary(BENEF_2008_CSV, BENEF_2008_PROC)
            preprocess_beneficiary(BENEF_2010_CSV, BENEF_2010_PROC)
            preprocess_claims()
            return {'status':'ok', 'message': 'Preprocessing completed.'}
        except Exception as e:
            return {'status': 'error', 'message': f'Preprocessing failed: {e}'}
    else:
        return {'status':'error', 'message': 'Preprocessing functions not available.'}

@app.post('/build-graph')
def build_graph_endpoint(): # Renamed
    if build_graph:
        if not os.path.exists(BENEF_2008_PROC) or not os.path.exists(BENEF_2010_PROC) or not os.path.exists(CLAIMS_PROC):
            return {'status': 'error', 'message': 'Processed data files not found. Run /preprocess first.'}
        try:
            build_graph()
            return {'status':'ok', 'message': 'Graph built successfully.'}
        except Exception as e:
            return {'status': 'error', 'message': f'Graph building failed: {e}'}
    else:
        return {'status':'error', 'message': 'Graph building function not available.'}

@app.post('/train')
def train_endpoint(): # Renamed
    if train_model:
        if not os.path.exists(GRAPH_PT):
            return {'status': 'error', 'message': 'Graph data not found. Run /build-graph first.'}
        try:
            # Consider running training in background? For now, it blocks.
            # You might want to adjust hyperparameters here or load from config
            train_model(epochs=50, lr=0.01, embedding_dim=32)
            return {'status':'ok', 'message': 'Training process completed.'}
        except Exception as e:
            return {'status': 'error', 'message': f'Training failed: {e}'}
    else:
        return {'status':'error', 'message': 'Training function not available.'}

@app.get('/anomalies')
def get_anomalies_endpoint(): # Renamed
    try:
        # Load model, graph, and mapping
        # Use the same embedding_dim as used in training
        model, data, reverse_node_mapping = load_model_and_graph(embedding_dim=32)

        if model and data and reverse_node_mapping:
            anomalies_list = score_anomalies(model, data, reverse_node_mapping, top_n=50) # Get top 50
            return {'anomalies': anomalies_list}
        else:
            # Errors during loading are printed in load_model_and_graph
            return {'status': 'error', 'message': 'Failed to load model or graph data. Check logs and ensure training was successful.', 'anomalies': []}
    except Exception as e:
        return {'status': 'error', 'message': f'Failed to get anomalies: {e}', 'anomalies': []}


if __name__ == '__main__':
    uvicorn.run("main:app", host='0.0.0.0', port=8000, reload=True) # Use string for reload