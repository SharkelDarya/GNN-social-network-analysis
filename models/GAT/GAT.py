import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.nn import ModuleList, Linear

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, heads=4, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.heads = heads
        self.hidden_channels = hidden_channels

        self.convs = ModuleList()

        if num_layers == 1:
            self.convs.append(
                GATConv(in_channels, hidden_channels, heads=heads, concat=False, dropout=dropout)
            )
        else:
            self.convs.append(
                GATConv(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout)
            )
            for _ in range(num_layers - 2):
                self.convs.append(
                    GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True, dropout=dropout)
                )
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=False, dropout=dropout)
            )

        self.lin = Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lin(x)           
        return x.view(-1)    
