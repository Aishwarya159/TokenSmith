import networkx as nx
from typing import List, Tuple

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_triplets(self, triplets: List[Tuple[str, str, str]]):
        for h, r, t in triplets:
            self.graph.add_edge(h, t, relation=r)

    def get_neighbors(self, entity: str, depth: int = 1):
        results = []
        visited = set()

        def dfs(node, d):
            if d == 0 or node in visited:
                return
            visited.add(node)

            for nbr in self.graph.neighbors(node):
                rel = self.graph[node][nbr]["relation"]
                results.append((node, rel, nbr))
                dfs(nbr, d - 1)

        dfs(entity, depth)
        return results
    def get_graph(self):
        return self.graph