import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch_geometric.utils import negative_sampling
from model import GCNEncoder, FraudGAE
from config import GRAPH_PT, MODEL_SAVE_PATH, MODEL_DIR, EMBEDDING_DIM, EPOCHS, LEARNING_RATE, PATIENCE

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train_model(epochs=200, lr=LEARNING_RATE, embedding_dim=EMBEDDING_DIM, patience=PATIENCE):
    """Trains the Graph Autoencoder model with early stopping and GPU support."""
    print("Starting model training...")

    try:
        data = torch.load(GRAPH_PT, weights_only=False)
        print(f"Loaded graph data from {GRAPH_PT}")
    except FileNotFoundError:
        print(f"Error: Graph data file not found at {GRAPH_PT}. Run graph building first.")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)
    data = data.to(device)

    in_channels = data.num_node_features
    encoder = GCNEncoder(in_channels, embedding_dim)
    model = FraudGAE(encoder).to(device)

    optimizer = Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    patience_counter = 0
    train_losses = []

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

        train_losses.append(loss.item())

        print(f"Epoch: {epoch+1:03d}, Loss: {loss:.4f}")

        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training complete. Best model saved to {MODEL_SAVE_PATH}")

    # Plot training curve
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GAE Training Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(MODEL_DIR, "training_curve.png"))
    print(f"Training curve saved to {os.path.join(MODEL_DIR, 'training_curve.png')}")


if __name__ == '__main__':
    train_model(epochs=EPOCHS, lr=LEARNING_RATE, embedding_dim=EMBEDDING_DIM)
