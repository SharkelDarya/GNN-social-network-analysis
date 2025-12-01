## Graph Neural Networks for Social Network Analysis

## Project Overview

This project explores how **Graph Neural Networks (GNNs)** can be used to analyze **social network structures and user behavior**. The main idea is to use modern graph-based machine learning models to understand how users are connected, how they interact, and what features influence their relationships.


Technologies: **Python 3, Pandas, NetworkX, Matplotlib, PyVis, PyTorch, PyTorch Geometric**


## Plan


1. Data Preparation

- [x] Loading and analyzing large social network datasets (e.g. Twitch dataset).
- [x] Cleaning and preprocessing user data and features.
- [x] Converting the data into a graph structure using NetworkX and PyTorch Geometric.

  \


2. Graph Construction & Visualization

- [x] Building a connected user subgraph for analysis.
- [x] Visualizing the graph using:
- [x] Matplotlib (static visualization with colors per language),
- [x] PyVis (interactive HTML graph visualization).

  \


3. GNN Model Implementation

- [ ] GCN (Graph Convolutional Network)


- [ ] GraphSAGE


- [ ] GAT (Graph Attention Network)

  \


4. Model Evaluation

- [ ] Comparing GNN performance with classical methods.


- [ ] Evaluating metrics such as accuracy, F1-score, and ROC AUC.


- [ ] Analyzing learned graph embeddings and node representations.


## Twitch Gamers Dataset

A social network of **Twitch users**, collected from the public API in **Spring 2018**. \n Nodes represent Twitch users, and edges represent **mutual follower relationships**. \n The graph forms a single strongly connected component and contains **no missing attributes**.


**Source (citation):** \n B. Rozemberczki and R. Sarkar. *Twitch Gamers: a Dataset for Evaluating Proximity Preserving and Structural Role-based Node Embeddings.* 2021. \n arXiv:2101.03091

```javascript
@misc{rozemberczki2021twitch,
  title={Twitch Gamers: a Dataset for Evaluating Proximity Preserving and Structural Role-based Node Embeddings}, 
  author={Benedek Rozemberczki and Rik Sarkar},
  year={2021},
  eprint={2101.03091},
  archivePrefix={arXiv},
  primaryClass={cs.SI}
}
```


