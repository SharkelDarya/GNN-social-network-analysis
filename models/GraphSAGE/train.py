import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from GraphSAGE import GraphSAGE
from torch_geometric.data import Data

# ---------------------------------------------------------
# Load prepared graph
# ---------------------------------------------------------
def load_graph(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# ---------------------------------------------------------
# Train GraphSAGE with early stopping
# ---------------------------------------------------------

def train_model(graph_path, epochs=500, lr=0.0097, hidden_channels=64, num_layers=2, dropout=0.324, patience=50):
    print("Loading graph...")
    obj = load_graph(graph_path)
    data: Data = obj['data']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    model = GraphSAGE(
        in_channels=data.x.size(1),
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    print("Start training...")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        train_loss = criterion(out[data.train_mask], data.y[data.train_mask])
        train_loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
            rmse_log = torch.sqrt(((out - data.y)**2).mean()).item()
            mae_log = (out - data.y).abs().mean().item()

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss.item():.4f} | "
              f"Val Loss: {val_loss.item():.4f} | RMSE_log: {rmse_log:.4f} | MAE_log: {mae_log:.4f}")

        if epochs_no_improve >= 1000:
            print(f"Early stopping triggered.")
            break

    # Save the best model
    if best_state:
        torch.save(best_state, 'models/GraphSAGE/GraphSAGE_best_model.pt')
        print("Best model saved to GraphSAGE_best_model.pt")

# ---------------------------------------------------------
# Запуск
# ---------------------------------------------------------
# (32, 4, 0.0, 0.01)
if __name__ == '__main__':
    train_model('dataset\processed_graph.pkl', epochs=500, lr=0.001, hidden_channels=32, num_layers=5, dropout=0.0, patience=20)
