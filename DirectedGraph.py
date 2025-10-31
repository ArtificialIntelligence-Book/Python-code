class DirectedGraph:
    def __init__(self):
        # Representation: adjacency list (node -> list of neighbors)
        self.graph = {}
    
    def add_node(self, node):
        if node not in self.graph:
            self.graph[node] = []
    
    def add_edge(self, from_node, to_node):
        # Add nodes if they don't exist
        self.add_node(from_node)
        self.add_node(to_node)
        # Add a directed edge from from_node to to_node
        self.graph[from_node].append(to_node)
    
    def get_neighbors(self, node):
        # Returns neighbors of node
        return self.graph.get(node, [])
    
    def __str__(self):
        return str(self.graph)


# Example usage:
g = DirectedGraph()
g.add_edge('A', 'B')
g.add_edge('A', 'C')
g.add_edge('B', 'C')
g.add_edge('C', 'A')
print("Directed Graph adjacency list:", g)