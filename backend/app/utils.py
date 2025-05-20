import torch
import json
from torch_geometric.data import Data
# from model import GCNEncoder, FraudGAE
# from config import GRAPH_PT, NODE_MAPPING_PATH, EMBEDDING_DIM, MODEL_SAVE_PATH
from .config import GRAPH_PT, NODE_MAPPING_PATH, EMBEDDING_DIM, MODEL_SAVE_PATH

from .model import GCNEncoder, FraudGAE


def load_graph():
    return torch.load(GRAPH_PT, weights_only=False)

def load_node_mapping():
    with open(NODE_MAPPING_PATH, 'r') as f:
        return json.load(f)

def get_node_index(claim_id, node_mapping):
    return node_mapping.get(str(claim_id))

def load_model(in_channels):
    encoder = GCNEncoder(in_channels, EMBEDDING_DIM)
    model = FraudGAE(encoder)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    return model

def process_claim_data(claim):
    """
    Process the claim data dictionary into the format expected by the model.
    
    Parameters:
    -----------
    claim : dict
        The raw claim data from the API request
        
    Returns:
    --------
    dict
        Processed claim data ready for the model
    """
    processed_data = {
        "patient_id": claim.patient_id,
        "provider_id": claim.provider_id,
        "claim_amount": float(claim.claim_amount),
        "claim_date": claim.claim_date,
        "patient_info": claim.patient_info,
        "provider_info": claim.provider_info,
        "claim_details": claim.claim_details
    }
    
    return processed_data
