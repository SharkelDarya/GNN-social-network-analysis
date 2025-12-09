import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch.nn import Linear, ModuleList

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.convs = ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        
        self.lin = Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.lin(x)
        return x.squeeze(-1) 