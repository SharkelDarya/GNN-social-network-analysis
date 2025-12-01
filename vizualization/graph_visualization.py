import pandas as pd
import torch
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import json

edges_df = pd.read_csv("data/large_twitch_edges.csv")  # numeric_id_1,numeric_id_2
nodes_df = pd.read_csv("data/large_twitch_features.csv")  # views,mature,life_time,created_at,updated_at,numeric_id,dead_account,language,affiliate

nodes_df = nodes_df[['numeric_id', 'language']]

# Create graph
G_full = nx.from_pandas_edgelist(edges_df, 'numeric_id_1', 'numeric_id_2')
language_dict = nodes_df.set_index('numeric_id')['language'].to_dict()
nx.set_node_attributes(G_full, language_dict, 'language')


num_nodes = 100

def get_connected_subgraph(G, num_nodes):
    start_node = next(iter(G.nodes))
    connected_nodes = set([start_node])
    frontier = set(G.neighbors(start_node))
    
    while len(connected_nodes) < num_nodes and frontier:
        node = frontier.pop()
        connected_nodes.add(node)
        frontier.update(set(G.neighbors(node)) - connected_nodes)
    
    return G.subgraph(connected_nodes)

subG = get_connected_subgraph(G_full, num_nodes)


with open("vizualization\language_colors.json", "r") as f:
    color_map_hex = json.load(f)


languages = list(set(nx.get_node_attributes(subG, 'language').values()))

def get_color(lang):
    return color_map_hex.get(lang, color_map_hex["OTHER"])

# HEX -> RGB
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4))

color_map = {lang: hex_to_rgb(get_color(lang)) for lang in languages}
node_colors = [color_map[subG.nodes[n]['language']] for n in subG.nodes()]

plt.figure(figsize=(10, 10))
pos = nx.kamada_kawai_layout(subG)
nx.draw(subG, pos, with_labels=False, node_color=node_colors, node_size=80, edge_color='black', alpha=1)

for lang, rgb in color_map.items():
    plt.scatter([], [], c=[rgb], label=lang)
plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title="Language")
plt.title("Graph with 100 users")
plt.axis('off')
plt.show()


# PyVis
net = Network(height='800px', width='1000px', bgcolor='#ffffff', font_color='black')
net.from_nx(subG)

for node in net.nodes:
    lang = subG.nodes[node['id']]['language']
    color_hex = get_color(lang)
    node['color'] = color_hex
    node['title'] = f"Language: {lang}"
    node['size'] = 20 + 5 * subG.degree(node['id'])

net.write_html("vizualization\connected_subgraph.html", open_browser=True)

# Data for PyTorch Geometric
node_id_mapping = {nid: i for i, nid in enumerate(subG.nodes())}
edge_index = torch.tensor([
    [node_id_mapping[src], node_id_mapping[dst]] 
    for src, dst in subG.edges()
], dtype=torch.long).t().contiguous()

language_to_idx = {lang: i for i, lang in enumerate(languages)}
x = torch.tensor([language_to_idx[subG.nodes[n]['language']] for n in subG.nodes()], dtype=torch.long)
x_one_hot = torch.nn.functional.one_hot(x, num_classes=len(languages)).float()

data = Data(x=x_one_hot, edge_index=edge_index)
print(data)
