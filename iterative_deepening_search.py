def iterative_deepening_search(graph, start, goal, max_depth=50):
    """
    Performs Iterative Deepening Search by calling DLS with increasing depth.
    
    :param graph: Graph adjacency list
    :param start: Start node
    :param goal: Goal node
    :param max_depth: Maximum depth limit to try
    :return: Path to goal if found, otherwise None
    """
    for depth in range(max_depth):
        print(f"Trying depth limit: {depth}")
        path = depth_limited_search(graph, start, goal, depth)
        if path:
            return path
    return None


print("Iterative Deepening Search:", iterative_deepening_search(graph_example, 'A', 'F'))