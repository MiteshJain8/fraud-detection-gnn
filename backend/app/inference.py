import torch
from torch_geometric.utils import k_hop_subgraph
from torch.nn.functional import sigmoid

def get_fraud_score(claim_id: str, data, z, node_mapping: dict, num_hops: int = 5, topk: int = 1):
    """
    Computes fraud risk score for a given claim using local k-hop neighborhood.

    Args:
        claim_id (str): Unique identifier for the claim.
        data (torch_geometric.data.Data): The graph data.
        z (torch.Tensor): Node embeddings from GNN.
        node_mapping (dict): Maps claim IDs to node indices.
        num_hops (int): Number of hops for subgraph neighborhood.
        topk (int): Number of top similar nodes to return.

    Returns:
        dict: Fraud score and top similar nodes with scores.
    """
    if claim_id not in node_mapping:
        raise ValueError(f"Claim ID {claim_id} not found in node mapping.")

    node_index = node_mapping[claim_id]

    # Extract k-hop subgraph around the node
    subset, _, _, _ = k_hop_subgraph(
        node_index, num_hops=num_hops,
        edge_index=data.edge_index, relabel_nodes=False
    )

    target_embedding = z[node_index]
    neighbor_embeddings = z[subset]

    # Dot product similarity + sigmoid
    scores = torch.matmul(neighbor_embeddings, target_embedding)
    sigmoid_scores = sigmoid(scores)
    fraud_score = sigmoid_scores.mean().item()

    top_scores, top_indices = torch.topk(sigmoid_scores, k=topk)
    top_neighbors = [
        {"node_index": int(subset[i]), "score": round(float(score), 4)}
        for i, score in zip(top_indices, top_scores)
    ]

    return {
        "claim_id": claim_id,
        "fraud_score": round(fraud_score, 4),
        "top_neighbors": top_neighbors
    }
