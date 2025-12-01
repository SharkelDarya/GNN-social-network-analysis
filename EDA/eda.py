import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

edges_df = pd.read_csv('data/large_twitch_edges.csv')
nodes_df = pd.read_csv('data/large_twitch_features.csv')

G = nx.Graph()
for _, row in tqdm(edges_df.iterrows(), total=len(edges_df), desc="Adding edges"):
    G.add_edge(row['numeric_id_1'], row['numeric_id_2'])

for node in tqdm(nodes_df['numeric_id'], desc="Adding missing nodes"):
    if node not in G:
        G.add_node(node)

# Nodes
with tqdm(total=1, desc="Counting nodes") as pbar:
    N = G.number_of_nodes()
    pbar.update(1)

# Edges
with tqdm(total=1, desc="Counting edges") as pbar:
    E = G.number_of_edges()
    pbar.update(1)

# Density
with tqdm(total=1, desc="Computing density") as pbar:
    density = nx.density(G)
    pbar.update(1)

# Transitivity
with tqdm(total=1, desc="Computing transitivity") as pbar:
    transitivity = nx.transitivity(G)
    pbar.update(1)

# Average clustering
with tqdm(total=1, desc="Computing average clustering") as pbar:
    avg_clustering = nx.average_clustering(G)
    pbar.update(1)


print(f"Nodes: {N}")
print(f"Edges: {E}")
print(f"Density: {density:.6f}")
print(f"Transitvity: {transitivity:.6f}")
print(f"Average clustering: {avg_clustering:.6f}")

degree_sequence = []
for _, d in tqdm(G.degree(), total=N, desc="Computing degrees"):
    degree_sequence.append(d)

degree_count = pd.Series(degree_sequence).value_counts().sort_index()

print("\nDistribution of degrees:")
print(degree_count)

plt.figure(figsize=(8,5))
plt.bar(degree_count.index, degree_count.values)
plt.xlabel('Node degree')
plt.ylabel('Amount of nodes')
plt.title('Distribution of degrees')
plt.show()
