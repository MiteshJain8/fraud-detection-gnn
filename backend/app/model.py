import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        # Use GCNConv layers for encoding
        # Adjust hidden layer size as needed
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        # Apply GCN layers with ReLU activation
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

# Define the Graph Autoencoder model using the GCNEncoder
# The GAE model uses the encoder to generate node embeddings
# and then computes reconstruction loss based on the dot product of embeddings.
class FraudGAE(GAE):
    def __init__(self, encoder):
        super(FraudGAE, self).__init__(encoder)

    # Override the encode method if custom logic is needed,
    # otherwise the base GAE encode method is used.
    # def encode(self, *args, **kwargs):
    #     return self.encoder(*args, **kwargs)

    # The decode method (inner product) is inherited from GAE
    # The recon_loss method (binary cross-entropy) is inherited from GAE
