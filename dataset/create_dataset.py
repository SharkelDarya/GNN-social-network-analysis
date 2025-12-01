import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch_geometric.data import Data
import pickle
from sklearn.model_selection import train_test_split

def load_edges(edge_file: str):
    edges = pd.read_csv(edge_file, header=0)
    edges = edges.applymap(lambda x: int(str(x).strip()))
    edge_index = torch.tensor(edges.values.T, dtype=torch.long)

    rev = edge_index[[1,0], :]
    edge_index = torch.cat([edge_index, rev], dim=1)
    return edge_index

# ---------------------------------------------------------
def load_features(node_file: str):
    df = pd.read_csv(node_file)

    df['views'] = df['views'].clip(lower=0)
    df['views_log'] = np.log1p(df['views'])
    y = torch.tensor(df['views_log'].values, dtype=torch.float)

    df['created_at'] = pd.to_datetime(df['created_at']).view('int64') // 10**9
    df['updated_at'] = pd.to_datetime(df['updated_at']).view('int64') // 10**9

    X = df.drop(columns=['views','views_log','numeric_id'])

    lang_enc = OneHotEncoder(sparse_output=False)
    lang = lang_enc.fit_transform(X[['language']])

    num_cols = ['mature','life_time','dead_account','affiliate','created_at','updated_at']
    scaler = StandardScaler()
    num_scaled = scaler.fit_transform(X[num_cols])

    features = torch.tensor(np.hstack((num_scaled, lang)), dtype=torch.float)
    return features, y, lang_enc, scaler

# ---------------------------------------------------------
def create_dataset(edge_file: str, node_file: str, out_file: str):
    edge_index = load_edges(edge_file)
    X, y, lang_enc, scaler = load_features(node_file)

    idx = np.arange(len(y))
    train_idx, temp_idx = train_test_split(idx, test_size=0.30, shuffle=True, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, shuffle=True, random_state=42)

    data = Data(x=X, edge_index=edge_index, y=y)
    data.train_mask = torch.zeros(len(y), dtype=torch.bool)
    data.val_mask = torch.zeros(len(y), dtype=torch.bool)
    data.test_mask = torch.zeros(len(y), dtype=torch.bool)

    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True

    with open(out_file, 'wb') as f:
        pickle.dump({'data': data, 'lang_enc': lang_enc, 'scaler': scaler}, f)

    print(f"Saved processed data to {out_file}")

# ---------------------------------------------------------
if __name__ == '__main__':
    create_dataset('data/large_twitch_edges.csv', 'data/large_twitch_features.csv', 'models/GCN/processed_graph.pkl')
