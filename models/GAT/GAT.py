import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, num_layers=2, dropout=0.3, heads=2):
        super().__init__()

        self.dropout = dropout
        self.heads = heads
        self.num_layers = num_layers
        self.hidden = hidden_channels

        self.convs = nn.ModuleList()

        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=heads, concat=False)
        )

        for _ in range(num_layers - 1):
            self.convs.append(
                GATConv(hidden_channels, hidden_channels, heads=heads, concat=False)
            )

        self.lin = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lin(x)
        return x.view(-1)
