def greedy_best_first_search(graph, start, goal, heuristic):
    """
    Greedy Best-First Search algorithm.
    
    :param graph: Graph adjacency list with costs (node -> [(neighbor, cost), ...])
    :param start: Start node
    :param goal: Goal node
    :param heuristic: Heuristic function to estimate distance to goal
    :return: Path to goal or None
    """
    from queue import PriorityQueue
    
    pq = PriorityQueue()
    pq.put((heuristic(start), start, [start]))
    visited = set()

    while not pq.empty():
        _, node, path = pq.get()
        if node == goal:
            return path
        visited.add(node)
        for neighbor, _ in graph.get(node, []):
            if neighbor not in visited:
                pq.put((heuristic(neighbor), neighbor, path + [neighbor]))
    return None


path = greedy_best_first_search(graph_with_costs, 'A', 'F', heuristic)
print("Greedy Best-First Search:", path)