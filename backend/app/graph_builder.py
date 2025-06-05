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
    NODE_MAPPING_PATH,
    PROCESSED_DIR
)

def build_graph():
    """
    Constructs a heterogeneous graph of beneficiaries and providers from preprocessed CSVs,
    encodes node features via one-hot demographic, chronic flags, normalized payments,
    and provider indicator. Saves PyG Data object and node mapping.
    """
    os.makedirs(os.path.dirname(GRAPH_PT), exist_ok=True)
    os.makedirs(os.path.dirname(NODE_MAPPING_PATH), exist_ok=True)

    try:
        ben8 = pd.read_csv(BENEF_2008_PROC, dtype=str)
        ben10 = pd.read_csv(BENEF_2010_PROC, dtype=str)
        claims = pd.read_csv(CLAIMS_PROC, dtype=str)
    except FileNotFoundError as e:
        print(f"Error loading processed files: {e} Run data_preprocess.py first.")
        return

    beneficiaries = pd.concat([ben8, ben10], ignore_index=True)
    beneficiaries.drop_duplicates(subset="DESYNPUF_ID", inplace=True)

    G = nx.Graph()
    node_mapping = {}
    current_idx = 0

    # === Demographic columns ===
    demogs = [
        "BENE_SEX_IDENT_CD",
        "BENE_RACE_CD",
        "BENE_ESRD_IND",
        "SP_STATE_CODE",
        "BENE_COUNTY_CD"
    ]

    # === Chronic condition flags ===
    chronic_flags = [
        "SP_ALZHDMTA", "SP_CHF", "SP_CHRNKIDN", "SP_CNCR", "SP_COPD",
        "SP_DEPRESSN", "SP_DIABETES", "SP_ISCHMCHT", "SP_OSTEOPRS",
        "SP_RA_OA", "SP_STRKETIA"
    ]

    # === Payment-related fields ===
    payment_fields = [
        "MEDREIMB_IP", "BENRES_IP", "PPPYMT_IP",
        "MEDREIMB_OP", "BENRES_OP", "PPPYMT_OP",
        "MEDREIMB_CAR", "BENRES_CAR", "PPPYMT_CAR"
    ]

    # Clean and encode features
    beneficiaries[chronic_flags + payment_fields] = beneficiaries[chronic_flags + payment_fields].fillna(0)
    beneficiaries[payment_fields] = beneficiaries[payment_fields].apply(pd.to_numeric, errors='coerce').fillna(0)
    beneficiaries[payment_fields] = (
        beneficiaries[payment_fields] - beneficiaries[payment_fields].mean()
    ) / (beneficiaries[payment_fields].std() + 1e-6)

    # One-hot encode demographics
    ben_dummies = pd.get_dummies(
        beneficiaries[demogs],
        columns=demogs,
        prefix=demogs,
        dummy_na=False
    )

    # Concatenate all features
    features_df = pd.concat(
        [ben_dummies.reset_index(drop=True),
        beneficiaries[chronic_flags + payment_fields].reset_index(drop=True)],
        axis=1
    )

    # Add +1 dimension for provider indicator
    feature_dim = features_df.shape[1] + 1
    num_benef = len(beneficiaries)
    provider_ids = claims["PRVDR_NUM"].dropna().unique()
    num_prov = len(provider_ids)
    total_nodes = num_benef + num_prov

    node_features = np.zeros((total_nodes, feature_dim), dtype=float)

    for i, row in beneficiaries.iterrows():
        pid = row["DESYNPUF_ID"]
        node_mapping[pid] = current_idx
        G.add_node(current_idx, node_type="beneficiary")
        node_features[current_idx, :-1] = features_df.iloc[i].values
        node_features[current_idx, -1] = 0  # beneficiary indicator
        current_idx += 1

    for pid in provider_ids:
        prov_key = str(pid)
        node_mapping[prov_key] = current_idx
        G.add_node(current_idx, node_type="provider")
        node_features[current_idx, :-1] = 0  # all other features zero
        node_features[current_idx, -1] = 1  # provider indicator
        current_idx += 1

    # === Edges ===
    edges = []
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
        G.add_edge(u, v)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.tensor(node_features, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)

    torch.save(data, GRAPH_PT)
    with open(NODE_MAPPING_PATH, 'w') as f:
        json.dump(node_mapping, f)

    print(f"Saved graph: {data.num_nodes} nodes, {data.num_edges} edges.")
    print(f"Node mapping at: {NODE_MAPPING_PATH}")
    with open(os.path.join(PROCESSED_DIR, 'feature_columns.json'), 'w') as f:
        json.dump(list(ben_dummies.columns), f)

if __name__ == "__main__":
    build_graph()
