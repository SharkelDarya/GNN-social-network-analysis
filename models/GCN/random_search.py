import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import random
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

        model = GCN(
            in_channels=x.size(1),
            hidden_channels=hidden,
            num_layers=layers,
            dropout=dropout
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.HuberLoss()

        best_val_loss = float('inf')
        best_state = None
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
                best_state = model.state_dict().copy()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                break

        return best_val_loss, best_state

    except RuntimeError as e:
        if 'out of memory' in str(e):
            torch.cuda.empty_cache()
            print("Skipping config due to CUDA OOM.")
        else:
            print(f"Skipping config due to runtime error: {e}")
        return None, None

# ---------------------------------------------------------
# Random search with unique configs
# ---------------------------------------------------------
def random_search(graph_path, n_iter=20):
    obj = load_graph(graph_path)
    data: Data = obj['data']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    hidden_options = [32, 64, 128, 256]
    layer_options = [1, 2, 3, 4]
    dropout_options = [0.0, 0.2, 0.5]
    lr_options = [0.001, 0.003, 0.005, 0.01]

    total_combinations = len(hidden_options) * len(layer_options) * len(dropout_options) * len(lr_options)
    n_iter = min(n_iter, total_combinations)

    if n_iter < 20:
        print(f"Warning: Only {total_combinations} unique combinations available. Setting n_iter to {total_combinations}")
        n_iter = total_combinations

    used_configs = set()

    all_configs = [(h, l, d, lr) for h in hidden_options
                                for l in layer_options
                                for d in dropout_options
                                for lr in lr_options]

    random.shuffle(all_configs)

    best_overall_loss = float('inf')
    best_overall_state = None
    best_config = None

    for i in range(1, n_iter + 1):
        hidden, layers, dropout, lr = all_configs[i-1]
        config_tuple = (hidden, layers, dropout, lr)

        print(f"Iteration {i}/{n_iter} | Config: hidden={hidden}, layers={layers}, "
              f"dropout={dropout}, lr={lr}")

        val_loss, state = train_one_config(
            data, hidden, layers, dropout, lr, epochs=50, patience=10, device=device
        )

        if val_loss is not None:
            used_configs.add(config_tuple)
            if val_loss < best_overall_loss:
                best_overall_loss = val_loss
                best_overall_state = state
                best_config = config_tuple
                print(f"New best config: {best_config} | Val Loss: {best_overall_loss:.4f}")
            else:
                print(f"Val Loss: {val_loss:.4f}")
    print(f"\nRandom search finished.")
    print(f"Total unique configs tested: {len(used_configs)}")
    print(f"Best config: {best_config} | Val Loss: {best_overall_loss:.4f}")

    return best_config, best_overall_loss, best_overall_state

# ---------------------------------------------------------
# Run
# ---------------------------------------------------------
if __name__ == '__main__':
    best_config, best_loss, best_state = random_search('/content/processed_graph.pkl', n_iter=400)