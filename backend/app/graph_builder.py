import torch
import networkx as nx
import pandas as pd
import numpy as np
import os
import json
from torch_geometric.data import Data
from app.config import (
    BENEF_2008_PROC,
    BENEF_2010_PROC,
    CLAIMS_PROC,
    GRAPH_PT,
    NODE_MAPPING_PATH,
)


def build_graph():
    """
    Constructs a heterogeneous graph of beneficiaries and providers from preprocessed CSVs,
    encodes node features via one-hot demographic and a provider indicator,
    builds edges for each claim, and saves a PyG Data object plus a node mapping JSON.
    """
    # Ensure output dirs
    os.makedirs(os.path.dirname(GRAPH_PT), exist_ok=True)
    os.makedirs(os.path.dirname(NODE_MAPPING_PATH), exist_ok=True)

    # Load preprocessed data
    try:
        ben8   = pd.read_csv(BENEF_2008_PROC, dtype=str)
        ben10  = pd.read_csv(BENEF_2010_PROC, dtype=str)
        claims = pd.read_csv(CLAIMS_PROC, dtype=str)
    except FileNotFoundError as e:
        print(f"Error loading processed files: {e} Run data_preprocess.py first.")
        return
    beneficiaries = pd.concat([ben8, ben10], ignore_index=True)
    beneficiaries.drop_duplicates(subset="DESYNPUF_ID", inplace=True)

    G = nx.Graph()
    node_mapping = {}
    current_idx = 0

    demogs = [
    "BENE_SEX_IDENT_CD",
    "BENE_RACE_CD",
    "BENE_ESRD_IND",
    "SP_STATE_CODE",
    "BENE_COUNTY_CD",
]

    # One-hot encode beneficiary demographics
    ben_dummies = pd.get_dummies(
        beneficiaries[demogs],
        columns=demogs,
        prefix=demogs,
        dummy_na=False
    )
    # Feature dim = demo dummies + provider indicator
    feature_dim = ben_dummies.shape[1] + 1
    num_benef = len(beneficiaries)
    provider_ids = claims["PRVDR_NUM"].dropna().unique()
    num_prov = len(provider_ids)
    total_nodes = num_benef + num_prov

    # Preallocate feature array
    node_features = np.zeros((total_nodes, feature_dim), dtype=float)

    for i, row in beneficiaries.iterrows():
        pid = row["DESYNPUF_ID"]
        node_mapping[pid] = current_idx
        G.add_node(current_idx, node_type="beneficiary")

        vec = ben_dummies.iloc[i].values
        node_features[current_idx, :-1] = vec
        # beneficiary indicator = 0
        node_features[current_idx, -1] = 0

        current_idx += 1

    for pid in provider_ids:
        prov_key = str(pid)
        node_mapping[prov_key] = current_idx
        G.add_node(current_idx, node_type="provider")
        # beneficiary dummies = 0, provider indicator = 1
        node_features[current_idx, -1] = 1
        current_idx += 1

    edges = []
    amounts = []
    for _, r in claims.iterrows():
        b_id = r["DESYNPUF_ID"]
        p_id = r["PRVDR_NUM"]
        if pd.isna(b_id) or pd.isna(p_id):
            continue
        u = node_mapping.get(str(b_id))
        v = node_mapping.get(str(p_id))
        if u is None or v is None:
            continue
        edges.append((u, v))
        amounts.append(float(r.get("CLM_PMT_AMT", 0)))
        G.add_edge(u, v)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.tensor(node_features, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    # data.edge_attr = torch.tensor(amounts, dtype=torch.float).unsqueeze(1)

    torch.save(data, GRAPH_PT)
    with open(NODE_MAPPING_PATH, 'w') as f:
        json.dump(node_mapping, f)

    print(f"Saved graph: {data.num_nodes} nodes, {data.num_edges} edges.")
    print(f"Node mapping at: {NODE_MAPPING_PATH}")

def build_subgraph_for_claim(claim_data, graph_data):
    """
    Build a subgraph for a specific claim by extracting relevant nodes and edges
    from the main graph.
    
    Parameters:
    -----------
    claim_data : dict
        The processed claim data containing patient_id and provider_id
    graph_data : torch_geometric.data.Data
        The main graph data object
        
    Returns:
    --------
    torch_geometric.data.Data
        A subgraph containing nodes and edges relevant to this claim
    """
    # Load node mapping
    try:
        with open(NODE_MAPPING_PATH, 'r') as f:
            node_mapping = json.load(f)
    except FileNotFoundError:
        print(f"Node mapping file not found at {NODE_MAPPING_PATH}")
        # Return a dummy subgraph for development
        num_nodes = 10
        x = torch.randn(num_nodes, graph_data.x.shape[1])  # Match feature dimensions
        edge_index = torch.randint(0, num_nodes, (2, 20))
        return Data(x=x, edge_index=edge_index)
    
    # Get node indices for patient and provider
    patient_id = claim_data.get("patient_id")
    provider_id = claim_data.get("provider_id")
    
    # If we can't find the nodes in our mapping, return a dummy graph
    if patient_id not in node_mapping or provider_id not in node_mapping:
        print(f"Patient or provider not found in graph: {patient_id}, {provider_id}")
        num_nodes = 10
        x = torch.randn(num_nodes, graph_data.x.shape[1])
        edge_index = torch.randint(0, num_nodes, (2, 20))
        return Data(x=x, edge_index=edge_index)
    
    # Get node indices
    patient_node = node_mapping[patient_id]
    provider_node = node_mapping[provider_id]
    
    # Extract k-hop neighborhood (k=2 for example)
    k = 2
    nodes_to_keep = set([patient_node, provider_node])
    
    # Simple BFS to get k-hop neighborhood
    current_nodes = nodes_to_keep.copy()
    for _ in range(k):
        next_nodes = set()
        for node in current_nodes:
            # Find all neighbors of this node
            # This is a simplified approach - in a real implementation,
            # you'd use graph_data.edge_index to find neighbors efficiently
            for i in range(graph_data.edge_index.shape[1]):
                if graph_data.edge_index[0, i].item() == node:
                    next_nodes.add(graph_data.edge_index[1, i].item())
                elif graph_data.edge_index[1, i].item() == node:
                    next_nodes.add(graph_data.edge_index[0, i].item())
        nodes_to_keep.update(next_nodes)
        current_nodes = next_nodes
    
    # Create node mapping from original to subgraph indices
    nodes_list = sorted(list(nodes_to_keep))
    subgraph_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(nodes_list)}
    
    # Extract features for these nodes
    x_sub = graph_data.x[nodes_list]
    
    # Extract edges between these nodes
    edge_mask = []
    new_edges = []
    for i in range(graph_data.edge_index.shape[1]):
        src = graph_data.edge_index[0, i].item()
        dst = graph_data.edge_index[1, i].item()
        if src in nodes_to_keep and dst in nodes_to_keep:
            edge_mask.append(i)
            # Remap node indices
            new_edges.append([subgraph_mapping[src], subgraph_mapping[dst]])
    
    if not new_edges:
        # If no edges found, create at least one edge between patient and provider
        new_edges = [[0, 1]]  # Assuming patient is first and provider is second
    
    edge_index_sub = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
    
    # Create subgraph
    subgraph = Data(x=x_sub, edge_index=edge_index_sub)
    
    return subgraph



if __name__ == "__main__":
    build_graph()