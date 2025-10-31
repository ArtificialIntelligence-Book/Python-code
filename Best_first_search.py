import heapq

def best_first_search(graph, start, goal, heuristic):
    visited = set()
    # Priority queue with (heuristic, node)
    queue = [(heuristic[start], start)]
    
    while queue:
        _, node = heapq.heappop(queue)
        
        if node == goal:
            print(f"Reached goal: {node}")
            return
        
        if node not in visited:
            print(node, end=" ")
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    heapq.heappush(queue, (heuristic[neighbor], neighbor))
    
    print("Goal not reachable from start")

# Example heuristic values (could represent estimated distance to goal):
heuristic = {
    'A': 10,
    'B': 8,
    'C': 5,
    'D': 7,
    'E': 3,
    'F': 0  # Goal node heuristic = 0
}

print("\nBest First Search from A to F:")
best_first_search(graph, 'A', 'F', heuristic)