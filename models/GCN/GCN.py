import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout):
        super().__init__()

        self.dropout = dropout
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(in_channels, hidden_channels))

        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_channels, hidden_channels))

        self.lin = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):

        for conv in self.layers:
            x = conv(x, edge_index)
            x = F.relu(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lin(x)
        return x.view(-1)