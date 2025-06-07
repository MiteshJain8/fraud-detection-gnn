import torch
from torch_geometric.utils import k_hop_subgraph
from torch.nn.functional import sigmoid
import torch.nn.functional as F
import pandas as pd

def compute_temp_embedding(claim, data, model, node_mapping):
    device = next(model.parameters()).device
    model.eval()
    expected_dim = 395

    # 1. Demographic encoding
    demo_df = pd.DataFrame([{
        "BENE_SEX_IDENT_CD": claim["bene_sex_ident_cd"],
        "BENE_RACE_CD": claim["bene_race_cd"],
        "BENE_ESRD_IND": claim["bene_esrd_ind"],
        "SP_STATE_CODE": claim["sp_state_code"],
        "BENE_COUNTY_CD": claim["bene_county_cd"]
    }])
    demo_onehot = pd.get_dummies(demo_df)
    demo_tensor = torch.tensor(demo_onehot.values, dtype=torch.float).squeeze(0)

    # 2. Chronic and Payment values
    chronic_vals = torch.tensor(list(claim["sp_conditions"].values()), dtype=torch.float)
    payment_vals = torch.tensor(list(claim["payments"].values()), dtype=torch.float)
    indicator = torch.tensor([0.], dtype=torch.float)  # Beneficiary flag

    # 3. Combine components into x_feat
    x_feat = torch.cat([demo_tensor, chronic_vals, payment_vals, indicator])

    # 4. Pad to match expected feature dimension
    if x_feat.shape[0] < expected_dim:
        padding = torch.zeros(expected_dim - x_feat.shape[0])
        x_feat = torch.cat([x_feat, padding])
    elif x_feat.shape[0] > expected_dim:
        x_feat = x_feat[:expected_dim]  # Truncate extra dims if any

    x_new = x_feat.unsqueeze(0).to(device)

    # 5. Add edge from new node to provider
    provider_id = claim["provider_id"]
    if provider_id not in node_mapping:
        raise ValueError(f"Provider ID {provider_id} not found in node mapping.")
    provider_idx = node_mapping[provider_id]

    x_full = torch.cat([data.x.to(device), x_new], dim=0)
    new_node_idx = x_full.size(0) - 1

    edge_index = data.edge_index.to(device)
    edge_to_provider = torch.tensor([[new_node_idx], [provider_idx]], dtype=torch.long, device=device)
    edge_from_provider = torch.tensor([[provider_idx], [new_node_idx]], dtype=torch.long, device=device)
    edge_index_full = torch.cat([edge_index, edge_to_provider, edge_from_provider], dim=1)

    # 6. Encode and calculate similarity
    with torch.no_grad():
        z_all = model.encode(x_full, edge_index_full)

    z_new = z_all[new_node_idx]
    z_existing = z_all[:-1]

    z_norm = F.normalize(z_existing, p=2, dim=1)
    z_new_norm = F.normalize(z_new, p=2, dim=0)
    scores = torch.matmul(z_norm, z_new_norm)

    fraud_score = torch.sigmoid(scores).mean().item()

    topk = 5
    top_vals, top_indices = torch.topk(scores, k=topk)
    top_neighbors = [
        {"node_index": int(i), "score": float(s)}
        for i, s in zip(top_indices, top_vals)
    ]

    return fraud_score, top_neighbors, z_new

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
