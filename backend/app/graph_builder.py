import torch
import networkx as nx
import pandas as pd
import numpy as np
import os
import json
from torch_geometric.data import Data
from config import (
    BENEF_2008_PROC,
    BENEF_2010_PROC,
    CLAIMS_PROC,
    GRAPH_PT,
    NODE_MAP_JSON,
)


def build_graph():
    """
    Constructs a heterogeneous graph of beneficiaries and providers from preprocessed CSVs,
    encodes node features via one-hot demographic and a provider indicator,
    builds edges for each claim, and saves a PyG Data object plus a node mapping JSON.
    """
    # Ensure output dirs
    os.makedirs(os.path.dirname(GRAPH_PT), exist_ok=True)
    os.makedirs(os.path.dirname(NODE_MAP_JSON), exist_ok=True)

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
    with open(NODE_MAP_JSON, 'w') as f:
        json.dump(node_mapping, f)

    print(f"Saved graph: {data.num_nodes} nodes, {data.num_edges} edges.")
    print(f"Node mapping at: {NODE_MAP_JSON}")


if __name__ == "__main__":
    build_graph()