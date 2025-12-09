import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torch_geometric.data import Data
import numpy as np
import matplotlib.pyplot as plt
from GCN import GCN
import random

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Make cuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Seed set to: {seed}")

# ---------------------------------------------------------
# Load graph
# ---------------------------------------------------------
def load_graph(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# ---------------------------------------------------------
# Train GCN
# ---------------------------------------------------------
def train_model(graph_path, epochs, lr, hidden_channels, num_layers, dropout):
    print("Setting seed...")
    set_seed(42)
    
    print("Loading graph...")
    obj = load_graph(graph_path)
    data: Data = obj['data']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    model = GCN(
        in_channels=data.x.size(1),
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    #Nailepsza opcja dla CGN narazie 
    optimizer = optim.RAdam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # from torch.optim import AdamW
    # optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # optimizer = optim.RAdam(model.parameters(), lr=lr, weight_decay=1e-4)

    criterion = torch.nn.SmoothL1Loss()

    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    grad_norms = []

    print("Start training...")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        train_loss = criterion(out[data.train_mask], data.y[data.train_mask])
        train_loss.backward()

        # Gradient clipping
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

        # Track gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        grad_norms.append(total_norm ** 0.5)

        optimizer.step()

        # ---------------------------------------------------------
        # Validation
        # ---------------------------------------------------------
        model.eval()
        with torch.no_grad():
            val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
            rmse_log = torch.sqrt(((out - data.y)**2).mean()).item()
            mae_log = (out - data.y).abs().mean().item()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | Train Loss: {train_loss.item():.4f} | "
                f"Val Loss: {val_loss.item():.4f} | RMSE_log: {rmse_log:.4f} | MAE_log: {mae_log:.4f}"
            )

        if epochs_no_improve >= 50:
            print("Early stopping triggered.")
            break

    # Save the best model
    if best_state:
        print("Best valid loss: ", best_val_loss)
        torch.save(best_state, 'models\GCN\GCN_best_model.pt')
        print("Best model saved to GCN_best_model.pt")

    return grad_norms


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------
if __name__ == '__main__':
    grad_norms = train_model(
        '/content/processed_graph.pkl',
        epochs=2000,
        lr=0.02,
        hidden_channels=64,
        num_layers=4,
        dropout=0.0
    )

    plt.figure(figsize=(8, 5))
    plt.plot(grad_norms)
    plt.xlabel("Epoka")
    plt.ylabel("Norma gradientu")
    plt.title("Zmiana normy gradientu podczas treningu")
    plt.grid(True)
    plt.show()
