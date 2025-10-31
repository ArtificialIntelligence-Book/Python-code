import heapq

def uniform_cost_search(graph, start, goal):
    """
    Uniform-Cost Search algorithm using a priority queue.
    
    :param graph: A dictionary where keys are nodes and values are lists of tuples (neighbor, cost)
    :param start: Starting node
    :param goal: Goal node
    :return: The lowest cost path from start to goal and its cost
    """
    # Priority queue: (cumulative_cost, current_node, path)
    pq = [(0, start, [start])]
    visited = set()

    while pq:
        cost, node, path = heapq.heappop(pq)

        # If goal is reached, return path and cost
        if node == goal:
            return path, cost
        
        if node in visited:
            continue
        visited.add(node)

        # Explore neighbors
        for neighbor, edge_cost in graph.get(node, []):
            if neighbor not in visited:
                heapq.heappush(pq, (cost + edge_cost, neighbor, path + [neighbor]))
    
    return None, float('inf')


graph_with_costs = {
    'A': [('B', 2), ('C', 5)],
    'B': [('D', 1), ('E', 3)],
    'C': [('F', 2)],
    'D': [], 'E': [('F', 1)], 'F': []
}

path, cost = uniform_cost_search(graph_with_costs, 'A', 'F')
print("Uniform-Cost Search:", path, "Cost:", cost)