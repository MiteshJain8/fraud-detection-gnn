import torch
from model import GCNEncoder, FraudGAE
from utils import load_graph, load_model, load_node_mapping
from inference import get_fraud_score
from config import MODEL_SAVE_PATH, EMBEDDING_DIM

def run_inference(claim_id: str):
    print(f"Running fraud inference for claim: {claim_id}")

    # Load graph and trained model
    data = load_graph()
    node_mapping = load_node_mapping()

    encoder = GCNEncoder(data.num_node_features, EMBEDDING_DIM)
    model = FraudGAE(encoder)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        result = get_fraud_score(claim_id, data, z, node_mapping)

    print(f"\nFraud risk score for claim {result['claim_id']}: {result['fraud_score']:.4f}")
    print(f"Top similar nodes:")
    for neighbor in result['top_neighbors']:
        print(f"Node {neighbor['node_index']} → score: {neighbor['score']}")

if __name__ == "__main__":
    run_inference(claim_id="002C9FB269ECB7C8")
