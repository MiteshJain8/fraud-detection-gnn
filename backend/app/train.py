import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from .model import GCNEncoder, FraudGAE
from .config import GRAPH_PT, MODEL_SAVE_PATH, MODEL_DIR
import os

def train_model(epochs=100, lr=0.01, embedding_dim=16):
    """Trains the Graph Autoencoder model."""
    print("Starting model training...")
    # Load graph data
    try:
        data = torch.load(GRAPH_PT)
        print(f"Loaded graph data from {GRAPH_PT}")
    except FileNotFoundError:
        print(f"Error: Graph data file not found at {GRAPH_PT}. Run graph building first.")
        return

    # Ensure model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Model setup
    in_channels = data.num_node_features
    encoder = GCNEncoder(in_channels, embedding_dim)
    model = FraudGAE(encoder)

    optimizer = Adam(model.parameters(), lr=lr)

    # --- Training Loop ---
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Encode nodes
        z = model.encode(data.x, data.edge_index)

        # Calculate reconstruction loss (using positive and negative edges)
        # GAE's recon_loss uses Binary Cross Entropy with Logits
        # It requires positive edge index and negative edge index
        pos_edge_index = data.edge_index
        # Sample negative edges (edges that do not exist in the graph)
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=pos_edge_index.size(1), # Sample as many negative edges as positive
            method='sparse'
        )

        loss = model.recon_loss(z, pos_edge_index, neg_edge_index)

        # Backpropagation
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}')

    # Save the trained model state dict
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Training complete. Model saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    # Example of running training directly
    # Adjust epochs, lr, embedding_dim as needed
    train_model(epochs=50, lr=0.01, embedding_dim=32)
