import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pickle
from collections import deque
import random
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from collections import deque


# ---------------------------------------------------------
# Load edges
# ---------------------------------------------------------
def load_edges(path):
    df = pd.read_csv(path, header=0, dtype=str)
    src = df.iloc[:, 0].astype(str).str.strip().astype(int)
    dst = df.iloc[:, 1].astype(str).str.strip().astype(int)

    edges = np.vstack([src, dst]).T

    nodes = np.unique(edges)
    id_map = {nid: i for i, nid in enumerate(nodes)}

    mapped_src = np.array([id_map[x] for x in src])
    mapped_dst = np.array([id_map[x] for x in dst])

    src_all = np.concatenate([mapped_src, mapped_dst])
    dst_all = np.concatenate([mapped_dst, mapped_src])

    edge_index = torch.tensor([src_all, dst_all], dtype=torch.long)

    num_nodes = len(nodes)
    adj = [[] for _ in range(num_nodes)]
    for s, d in zip(src_all, dst_all):
        adj[s].append(int(d))

    return edge_index, adj, num_nodes, id_map, nodes


# ---------------------------------------------------------
# Graph features
# ---------------------------------------------------------
def compute_graph_features(edge_index, adj, num_nodes):
    print("degree...")
    deg = degree(edge_index[0], num_nodes=num_nodes).unsqueeze(1)

    print("pagerank...")
    pr = pagerank(edge_index, num_nodes)

    print("clustering (approx)...")
    clust = clustering(adj, num_nodes, samples=20000)

    # print("betweenness (approx)...")
    # bet = betweenness(adj, num_nodes, samples=1500)

    return torch.cat([deg, pr, clust], dim=1)


def pagerank(edge_index, num_nodes, alpha=0.85, iters=30):
    src, dst = edge_index  # src -> dst
    out_deg = degree(src, num_nodes=num_nodes)   # out-degree of source
    out_inv = 1.0 / (out_deg + 1e-9)

    pr = torch.ones(num_nodes) / num_nodes
    teleport = (1 - alpha) / num_nodes

    for _ in range(iters):
        new_pr = torch.zeros(num_nodes)
        # propagate from src -> dst
        new_pr.index_add_(0, dst, pr[src] * out_inv[src])
        pr = alpha * new_pr + teleport

    return pr.unsqueeze(1)



def clustering(adj, num_nodes, samples=20000):
    rng = random.Random(42)
    closed = np.zeros(num_nodes, dtype=np.int32)
    total = np.zeros(num_nodes, dtype=np.int32)

    # prepare neighbor sets for O(1) lookup
    neigh_sets = [set(nbrs) for nbrs in adj]

    for _ in range(samples):
        v = rng.randrange(num_nodes)
        neigh = list(neigh_sets[v])
        if len(neigh) < 2:
            continue
        a, b = rng.sample(neigh, 2)
        total[v] += 1
        if b in neigh_sets[a]:
            closed[v] += 1

    # avoid division by zero
    cluster = np.zeros(num_nodes, dtype=float)
    mask = total > 0
    cluster[mask] = closed[mask] / total[mask]

    return torch.tensor(cluster, dtype=torch.float32).unsqueeze(1)

def single_brandes(adj, num_nodes, seed):
    rng = random.Random(seed)
    s = rng.randrange(num_nodes)

    # adjacency as list of lists
    stack = []
    preds = [[] for _ in range(num_nodes)]
    sigma = np.zeros(num_nodes, dtype=np.float64)
    dist = -np.ones(num_nodes, dtype=np.int32)

    sigma[s] = 1.0
    dist[s] = 0
    q = deque([s])

    # BFS to get shortest paths DAG
    while q:
        v = q.popleft()
        stack.append(v)
        for w in adj[v]:
            if dist[w] < 0:
                dist[w] = dist[v] + 1
                q.append(w)
            if dist[w] == dist[v] + 1:
                sigma[w] += sigma[v]
                preds[w].append(v)

    # accumulation
    delta = np.zeros(num_nodes, dtype=np.float64)
    while stack:
        w = stack.pop()
        for v in preds[w]:
            delta_v = (sigma[v] / (sigma[w] + 1e-30)) * (1.0 + delta[w])
            delta[v] += delta_v
        if w != s:
            pass
        
    return delta  # numpy array length num_nodes


def betweenness(adj, num_nodes, samples=1500):
    bet = np.zeros(num_nodes, dtype=np.float64)
    seeds = list(range(samples))
    with Pool(cpu_count()) as pool:
        for delta in tqdm(pool.imap(partial(single_brandes, adj, num_nodes), seeds), total=samples):
            bet += delta
    # standardize
    bet = (bet - bet.mean()) / (bet.std() + 1e-9)
    return torch.tensor(bet, dtype=torch.float32).unsqueeze(1)



# ---------------------------------------------------------
# Load node features
# ---------------------------------------------------------
def load_node_features(node_file, num_nodes, id_map, nodes, max_views=500000):
    df = pd.read_csv(node_file)

    if "numeric_id" in df.columns:
        df.index = df["numeric_id"].map(id_map)
        df = df.sort_index()
    else:
        df = df.iloc[:num_nodes]

    df["views"] = df["views"].clip(upper=max_views)

    df["views_log"] = np.log1p(df["views"])
    y = torch.tensor(df["views_log"].values, dtype=torch.float32)

    df["created_at"] = pd.to_datetime(df["created_at"]).astype("int64") // 10**9
    df["updated_at"] = pd.to_datetime(df["updated_at"]).astype("int64") // 10**9

    X = df.drop(columns=["views", "views_log", "numeric_id"], errors="ignore")

    # One-hot lang
    lang_enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    lang = lang_enc.fit_transform(X[["language"]])

    # numerical
    num_cols = ["mature", "life_time", "dead_account", "affiliate", "created_at", "updated_at"]
    scaler = StandardScaler()
    num_scaled = scaler.fit_transform(X[num_cols])

    base_features = torch.tensor(np.hstack((num_scaled, lang)), dtype=torch.float32)

    return base_features, y, lang_enc, scaler

# ---------------------------------------------------------
# Create dataset
# ---------------------------------------------------------
def create_dataset(edge_file, node_file, out_file):
    edge_index, adj, num_nodes, id_map, nodes = load_edges(edge_file)

    # ------------------------------
    # Base node features
    # ------------------------------
    X_base, y, lang_enc, scaler_base = load_node_features(node_file, num_nodes, id_map, nodes)

    # ------------------------------
    # Graph features
    # ------------------------------
    print("Computing graph features...")
    X_graph = compute_graph_features(edge_index, adj, num_nodes)

    scaler_graph = StandardScaler()
    X_graph_scaled = torch.tensor(scaler_graph.fit_transform(X_graph), dtype=torch.float32)

    for i in range(X_graph_scaled.shape[1]):
        corr = np.corrcoef(X_graph_scaled[:, i], y.numpy())[0, 1]
        print(f"Graph feature {i} corr with target: {corr:.4f}")

    X_final = torch.cat([X_base, X_graph_scaled], dim=1)


    idx = np.arange(len(y))
    train_idx, temp_idx = train_test_split(idx, test_size=0.3, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    data = Data(x=X_final, edge_index=edge_index, y=y)
    data.train_mask = torch.zeros(len(y), dtype=torch.bool)
    data.val_mask = torch.zeros(len(y), dtype=torch.bool)
    data.test_mask = torch.zeros(len(y), dtype=torch.bool)

    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True

    with open(out_file, "wb") as f:
        pickle.dump({
            "data": data,
            "lang_enc": lang_enc,
            "scaler_base": scaler_base,
            "scaler_graph": scaler_graph
        }, f)

    print("Saved:", out_file)


# ---------------------------------------------------------
if __name__ == "__main__":
    create_dataset(
        "data/large_twitch_edges.csv",
        "data/large_twitch_features.csv",
        "dataset/processed_graph.pkl"
    )
