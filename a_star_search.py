def a_star_search(graph, start, goal, heuristic):
    """
    A* Search algorithm.
    
    :param graph: Graph adjacency list with costs (node -> [(neighbor, cost), ...])
    :param start: Start node
    :param goal: Goal node
    :param heuristic: A function estimating cost from node to goal
    :return: Path and cost if found, else None
    """
    pq = [(heuristic(start), 0, start, [start])]
    visited = set()

    while pq:
        est_total_cost, cost_so_far, node, path = heapq.heappop(pq)

        if node == goal:
            return path, cost_so_far
        
        if node in visited:
            continue
        visited.add(node)

        for neighbor, edge_cost in graph.get(node, []):
            if neighbor not in visited:
                g = cost_so_far + edge_cost
                f = g + heuristic(neighbor)
                heapq.heappush(pq, (f, g, neighbor, path + [neighbor]))
                
    return None, float('inf')


# Example heuristic: straight-line distance estimate (dummy heuristic)
heuristic_map = {
    'A': 7, 'B': 6, 'C': 2, 'D': 1, 'E': 0, 'F': 0
}

def heuristic(node):
    return heuristic_map.get(node, 0)

path, cost = a_star_search(graph_with_costs, 'A', 'F', heuristic)
print("A* Search:", path, "Cost:", cost)