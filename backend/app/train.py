import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from model import GCNEncoder, FraudGAE
from config import GRAPH_PT, MODEL_SAVE_PATH, MODEL_DIR, EMBEDDING_DIM

import os

def train_model(epochs=100, lr=0.01, embedding_dim=16):
    print("Starting model training...")
    try:
        data = torch.load(GRAPH_PT, weights_only=False)
        print(f"Loaded graph data from {GRAPH_PT}")
    except FileNotFoundError:
        print(f"Error: Graph data file not found at {GRAPH_PT}. Run graph building first.")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)

    in_channels = data.num_node_features
    encoder = GCNEncoder(in_channels, embedding_dim)
    model = FraudGAE(encoder)

    optimizer = Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)
        pos_edge_index = data.edge_index
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=pos_edge_index.size(1),
            method='sparse'
        )
        loss = model.recon_loss(z, pos_edge_index, neg_edge_index)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}')

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Training complete. Model saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train_model(epochs=50, lr=0.01, embedding_dim=EMBEDDING_DIM)
