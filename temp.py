#!/usr/bin/env python3

from operator import pos
import pickle
from src.knowledge_graph import KnowledgeGraph
with open("index/sections/textbook_index_kg.pkl", "rb") as f:
    G = pickle.load(f).get_graph()
import pickle

print(type(G))
print(G.number_of_nodes(), "nodes")
print(G.number_of_edges(), "edges")
print(list(G.nodes())[:5])
import networkx as nx
import matplotlib.pyplot as plt

# edge_labels = nx.get_edge_attributes(G, 'relation')
# pos = nx.spring_layout(G, k=0.5)

# nx.draw(G, pos, with_labels=True)
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

# plt.show()
adj_list = {}

for u, v, data in G.edges(data=True):
    relation = data.get("relation", "N/A")
    adj_list.setdefault(u, []).append((v, relation))

for node, neighbors in adj_list.items():
    print(f"{node}:")
    for v, relation in neighbors:
        print(f"  -> {v} ({relation})")