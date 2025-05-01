import torch
import networkx as nx
import pandas as pd
import numpy as np
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
from config import BENEF_2008_PROC, BENEF_2010_PROC, CLAIMS_PROC, GRAPH_PT, NODE_MAP_JSON # Added NODE_MAP_JSON
import os
import json

# Function to create one-hot encodings or other features
def create_features(df, columns, prefix):
    # Simple one-hot encoding for categorical features
    features = pd.get_dummies(df[columns], columns=columns, prefix=prefix, dummy_na=False)
    return torch.tensor(features.values, dtype=torch.float)

# merge 2008 & 2010 beneficiaries
def build_graph():
    # Ensure the processed directory exists
    os.makedirs(os.path.dirname(GRAPH_PT), exist_ok=True)
    os.makedirs(os.path.dirname(NODE_MAP_JSON), exist_ok=True)

    try:
        ben8 = pd.read_csv(BENEF_2008_PROC)
        ben10= pd.read_csv(BENEF_2010_PROC)
        claims = pd.read_csv(CLAIMS_PROC)
    except FileNotFoundError as e:
        print(f"Error loading processed files: {e}")
        print("Please run the preprocessing script first (e.g., python backend/app/data_preprocess.py)")
        return

    G = nx.Graph()
    node_mapping = {} # Map original IDs to sequential integer IDs
    node_features = []
    current_id = 0

    # Add beneficiary nodes
    beneficiaries = pd.concat([ben8, ben10]).drop_duplicates('DESYNPUF_ID')
    # Define beneficiary feature columns (example)
    beneficiary_feature_cols = ['BENE_SEX_IDENT_CD', 'BENE_RACE_CD', 'BENE_AGE_CAT_CD', 'BENE_STATE_CD'] # Add chronic conditions if needed
    # Handle potential missing values before creating features
    beneficiaries[beneficiary_feature_cols] = beneficiaries[beneficiary_feature_cols].fillna(-1) # Or use another strategy

    for _, r in beneficiaries.iterrows():
        original_id = str(r.DESYNPUF_ID)
        if original_id not in node_mapping:
            node_mapping[original_id] = current_id
            G.add_node(current_id, node_type='beneficiary') # Store original ID or type if needed
            # Create features for this node (example: one-hot encode categorical)
            # This part needs refinement based on how features are structured
            # For simplicity, let's create placeholder features first
            # features = create_features(r.to_frame().T, beneficiary_feature_cols, 'ben') # Simplified, needs proper feature engineering
            # node_features.append(features.squeeze().numpy()) # Append features
            current_id += 1

    # Add provider nodes
    provider_ids = claims.PRVDR_NUM.dropna().unique()
    for prv in provider_ids:
        original_id = str(prv)
        if original_id not in node_mapping:
            node_mapping[original_id] = current_id
            G.add_node(current_id, node_type='provider')
            # Add provider features (e.g., placeholder or based on claims data)
            # node_features.append(np.zeros(len(beneficiary_feature_cols))) # Placeholder, match feature dim
            current_id += 1

    # --- Feature Engineering (Refined Approach) ---
    # Create features *after* all nodes are added and mapped
    num_nodes = current_id
    # Example: Determine feature size based on beneficiary features + provider indicator
    # This is a simplified example. Real-world features would be more complex.
    ben_one_hot = pd.get_dummies(beneficiaries[beneficiary_feature_cols], columns=beneficiary_feature_cols, prefix='ben', dummy_na=False)
    feature_dim = ben_one_hot.shape[1] + 1 # +1 for provider indicator

    node_features_np = np.zeros((num_nodes, feature_dim))

    # Populate features for beneficiaries
    for _, r in beneficiaries.iterrows():
        original_id = str(r.DESYNPUF_ID)
        node_idx = node_mapping[original_id]
        ben_features_df = pd.get_dummies(r[beneficiary_feature_cols].to_frame().T, columns=beneficiary_feature_cols, prefix='ben', dummy_na=False)
        # Align columns with the full one-hot matrix
        ben_features_aligned = ben_features_df.reindex(columns=ben_one_hot.columns, fill_value=0)
        node_features_np[node_idx, :-1] = ben_features_aligned.values.squeeze()
        # Beneficiary indicator = 0
        node_features_np[node_idx, -1] = 0

    # Populate features for providers
    for prv in provider_ids:
        original_id = str(prv)
        node_idx = node_mapping[original_id]
        # Provider features: Set beneficiary features to 0, provider indicator to 1
        node_features_np[node_idx, :-1] = 0
        node_features_np[node_idx, -1] = 1

    node_features_tensor = torch.tensor(node_features_np, dtype=torch.float)
    # --- End Feature Engineering ---

    # Add edges using the mapped integer IDs
    edge_list = []
    edge_attributes = {'amount': [], 'claim_id': []} # Example edge attributes
    for _, r in claims.iterrows():
        beneficiary_original_id = str(r.DESYNPUF_ID)
        provider_original_id = str(r.PRVDR_NUM)

        if pd.notna(provider_original_id) and beneficiary_original_id in node_mapping and provider_original_id in node_mapping:
            u = node_mapping[beneficiary_original_id]
            v = node_mapping[provider_original_id]
            if G.has_node(u) and G.has_node(v):
                G.add_edge(u, v) # Add edge using integer IDs
                edge_list.append((u, v))
                # Add edge features if needed
                edge_attributes['amount'].append(r.CLM_PMT_AMT)
                edge_attributes['claim_id'].append(r.CLM_ID)

    # Convert NetworkX graph to PyTorch Geometric data object
    # Manually create edge_index
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # Create PyG Data object
    data = Data(x=node_features_tensor, edge_index=edge_index)
    # Add other attributes if needed (e.g., edge_attr)
    # data.edge_attr = torch.tensor(edge_attributes['amount'], dtype=torch.float).unsqueeze(1) # Example

    # Save the PyG data object and node mapping
    torch.save(data, GRAPH_PT)
    with open(NODE_MAP_JSON, 'w') as f:
        json.dump(node_mapping, f)

    print(f"Graph data saved to {GRAPH_PT}")
    print(f"Node mapping saved to {NODE_MAP_JSON}")
    print(f"Graph has {data.num_nodes} nodes and {data.num_edges} edges.")
    print(f"Node feature dimension: {data.num_node_features}")

if __name__ == '__main__':
    build_graph()