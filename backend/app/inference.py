import torch
from torch_geometric.utils import k_hop_subgraph
from torch.nn.functional import sigmoid
import pandas as pd
import json
import os
from config import PROCESSED_DIR

def compute_temp_embedding(data, model, claim: dict, node_mapping: dict):
    # Load training-time feature columns (from graph_builder.py)
    with open(os.path.join(PROCESSED_DIR, 'feature_columns.json')) as f:
        feature_cols = json.load(f)

    demo = {
    'BENE_SEX_IDENT_CD': claim['bene_sex_ident_cd'],
    'BENE_RACE_CD': claim['bene_race_cd'],
    'BENE_ESRD_IND': claim['bene_esrd_ind'],
    'SP_STATE_CODE': claim['sp_state_code'],
    'BENE_COUNTY_CD': claim['bene_county_cd']
}
    df_demo = pd.DataFrame([demo])
    demo_onehot = pd.get_dummies(df_demo, columns=df_demo.columns)
    demo_onehot = demo_onehot.reindex(columns=feature_cols, fill_value=0)
    demo_tensor = torch.tensor(demo_onehot.astype('float32').values, dtype=torch.float32).squeeze()

    # Add chronic condition flags
    chronic_vals = torch.tensor(list(claim['sp_conditions'].values()), dtype=torch.float32)

    # Add payment fields
    payment_vals = torch.tensor(list(claim['payments'].values()), dtype=torch.float32)

    # Add final provider indicator = 0
    indicator = torch.tensor([0.0], dtype=torch.float32)

    # Final feature vector
    new_node_feat = torch.cat([demo_tensor, chronic_vals, payment_vals, indicator])  # shape: [F]

    # Append to data.x
    new_x = torch.cat([data.x, new_node_feat.unsqueeze(0)], dim=0)
    
    # Resolve provider ID to node index
    provider_id = str(claim['provider_id'])
    if provider_id not in node_mapping:
        raise ValueError(f"Provider ID '{provider_id}' not found in training graph.")
    provider_idx = node_mapping[provider_id]
    new_node_idx = new_x.size(0) - 1  # index of appended node

    # Add edge between new node and provider
    new_edge = torch.tensor([[new_node_idx], [provider_idx]], dtype=torch.long)  # shape: [2, 1]
    new_edge_index = torch.cat([data.edge_index, new_edge], dim=1)

    # Encode full graph with new node
    model.eval()
    with torch.no_grad():
        z_all = model.encode(new_x, new_edge_index)
        z_new = z_all[new_node_idx]

        # Compare with 1-hop neighbors only to reduce memory
        subset, _, _, _ = k_hop_subgraph(new_node_idx, num_hops=1, edge_index=new_edge_index, relabel_nodes=False)
        neighbors = z_all[subset]
        sim_scores = torch.matmul(neighbors, z_new)
        fraud_score = torch.sigmoid(sim_scores).mean().item()

        # Get top similar neighbors
        topk = 5
        top_scores, top_indices = torch.topk(sim_scores, k=min(topk, len(sim_scores)))
        top_neighbors = [{"node_index": int(subset[i]), "score": float(s)} for i, s in zip(top_indices, top_scores)]

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
