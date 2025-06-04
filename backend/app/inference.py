import torch
from torch_geometric.utils import k_hop_subgraph
from torch.nn.functional import sigmoid
import pandas as pd

def compute_temp_embedding(claim_dict, z, data, model, edge_index):
    """
    Build temporary feature vector for new claim, connect to existing provider,
    and return the encoded node embedding.
    """
    # Extract components
    sex = claim_dict["bene_sex_ident_cd"]
    race = claim_dict["bene_race_cd"]
    esrd = claim_dict["bene_esrd_ind"]
    state = claim_dict["sp_state_code"]
    county = claim_dict["bene_county_cd"]
    conditions = claim_dict["sp_conditions"]
    payments = claim_dict["payments"]
    provider_id = claim_dict["provider_id"]

    # Build dummy DataFrame for one-hot
    df_demo = pd.DataFrame([{
        "BENE_SEX_IDENT_CD": sex,
        "BENE_RACE_CD": race,
        "BENE_ESRD_IND": esrd,
        "SP_STATE_CODE": state,
        "BENE_COUNTY_CD": county
    }])

    demo_onehot = pd.get_dummies(df_demo, columns=df_demo.columns)
    demo_onehot = demo_onehot.reindex(columns=data.x.shape[1]-1, fill_value=0)

    # Prepare chronic + payment features
    chronic_vals = torch.tensor([conditions[k] for k in sorted(conditions)], dtype=torch.float)
    payment_vals = torch.tensor([payments[k] for k in sorted(payments)], dtype=torch.float)
    payment_vals = (payment_vals - payment_vals.mean()) / (payment_vals.std() + 1e-6)

    node_feat = torch.cat([torch.tensor(demo_onehot.values, dtype=torch.float).squeeze(), chronic_vals, payment_vals, torch.tensor([0.])])  # 0 = beneficiary

    node_feat = node_feat.to(data.x.device)

    # Manually run through encoder
    return model.encoder(node_feat.unsqueeze(0), edge_index)


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
