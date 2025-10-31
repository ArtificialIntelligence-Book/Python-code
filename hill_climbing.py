def hill_climbing(start, neighbors_fn, heuristic):
    """
    Hill Climbing algorithm.
    
    :param start: Initial state
    :param neighbors_fn: Function that returns list of neighbors of a state
    :param heuristic: Function that returns heuristic value of a state (smaller is better)
    :return: Local optimum state found
    """
    current = start
    while True:
        neighbors = neighbors_fn(current)
        if not neighbors:
            break
        
        # Find neighbor with best (lowest) heuristic value
        neighbor = min(neighbors, key=heuristic)
        
        if heuristic(neighbor) >= heuristic(current):
            # No better neighbor found, local optimum reached
            break
        current = neighbor
    
    return current


# Example: Maximize negative values (minimize heuristic) in a function y = (x-3)^2

def neighbors_fn(x):
    # Small neighbors: x+1 and x-1
    return [x - 1, x + 1]

def heuristic(x):
    return (x - 3) ** 2

result = hill_climbing(0, neighbors_fn, heuristic)
print("Hill Climbing result:", result)