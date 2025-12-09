import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import random
from GraphSAGE import GraphSAGE
from torch_geometric.data import Data
from tqdm import tqdm

# ---------------------------------------------------------
# Load graph
# ---------------------------------------------------------
def load_graph(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# ---------------------------------------------------------
# Train
# ---------------------------------------------------------
def train_one_config(data, hidden, layers, dropout, lr, epochs=50, patience=20, device='cuda'):
    try:
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        y = data.y.to(device)
        train_mask = data.train_mask.to(device)
        val_mask = data.val_mask.to(device)

        model = GraphSAGE(
            in_channels=x.size(1),
            hidden_channels=hidden,
            num_layers=layers,
            dropout=dropout
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in tqdm(range(1, epochs + 1), desc="Epochs", leave=False):
            model.train()
            optimizer.zero_grad()
            out = model(x, edge_index)
            train_loss = criterion(out[train_mask], y[train_mask])
            train_loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = criterion(out[val_mask], y[val_mask])

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                break

        return best_val_loss

    except RuntimeError as e:
        if 'out of memory' in str(e):
            torch.cuda.empty_cache()
            print("Skipping config due to CUDA OOM.")
        else:
            print(f"Skipping config due to runtime error: {e}")
        return None, None

# ---------------------------------------------------------
# Random search
# ---------------------------------------------------------
def random_search(graph_path, n_iter=20):
    obj = load_graph(graph_path)
    data: Data = obj['data']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    hidden_options = [32, 64, 128]
    layer_options = [2, 3, 4]
    dropout_options = [0.0, 0.2, 0.5]
    lr_options = [0.001, 0.003, 0.005, 0.01]

    best_overall_loss = float('inf')
    best_config = None

    for i in range(1, n_iter + 1):
        hidden = random.choice(hidden_options)
        layers = random.choice(layer_options)
        dropout = random.choice(dropout_options)
        lr = random.choice(lr_options)

        print(f"Iteration {i}/{n_iter} | Config: hidden={hidden}, layers={layers}, "
              f"dropout={dropout}, lr={lr}")

        val_loss = train_one_config(
            data, hidden, layers, dropout, lr, epochs=50, patience=10, device=device
        )

        if val_loss is not None and val_loss < best_overall_loss:
            best_overall_loss = val_loss
            best_config = (hidden, layers, dropout, lr)
            print(f"New best config: {best_config} | Val Loss: {best_overall_loss:.4f}")

    print(f"Random search finished. Best config: {best_config} | Val Loss: {best_overall_loss:.4f}")

# ---------------------------------------------------------
# Run
# ---------------------------------------------------------
if __name__ == '__main__':
    random_search('dataset\processed_graph.pkl', n_iter=20)
