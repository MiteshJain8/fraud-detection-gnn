import os
import torch
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch_geometric.utils import negative_sampling
from model import GCNEncoder, FraudGAE
from config import GRAPH_PT, MODEL_SAVE_PATH, MODEL_DIR, EMBEDDING_DIM, EPOCHS, LEARNING_RATE, PATIENCE
from sklearn.metrics import roc_auc_score

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

        pos_pred = model.decoder(z, pos_edge_index)
        neg_pred = model.decoder(z, neg_edge_index)

        # Concatenate
        logits = torch.cat([pos_pred, neg_pred], dim=0)
        labels = torch.cat([
            torch.ones(pos_pred.size(0), device=logits.device),
            torch.zeros(neg_pred.size(0), device=logits.device)
        ], dim=0)

        # Define class weights (inverse of frequency)
        pos_weight = torch.tensor([10.0], device=logits.device)  # can tune this
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        print(f"Epoch: {epoch+1:03d}, Loss: {loss:.4f}")

        model.eval()
        with torch.no_grad():
            z = model.encode(data.x, data.edge_index)

            # Sample test edges or use val set
            scores = model.decoder(z, pos_edge_index).sigmoid()
            labels = torch.ones_like(scores)

            scores_neg = model.decoder(z, neg_edge_index).sigmoid()
            labels_neg = torch.zeros_like(scores_neg)

            all_scores = torch.cat([scores, scores_neg]).cpu().numpy()
            all_labels = torch.cat([labels, labels_neg]).cpu().numpy()

            auc = roc_auc_score(all_labels, all_scores)
            precision_at_k = (all_scores.argsort()[::-1][:100] < len(scores)).sum() / 100.0

            print(f"Epoch {epoch+1}, AUC: {auc:.4f}, Precision@100: {precision_at_k:.4f}")

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
