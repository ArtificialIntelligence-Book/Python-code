import math
import random

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0

    def ucb1(self, total_visits, c=1.4):
        if self.visits == 0:
            return float('inf')  # Prioritize unvisited nodes
        return self.wins / self.visits + c * math.sqrt(math.log(total_visits) / self.visits)

def mcts(root, simulate_fn, itermax=1000):
    for _ in range(itermax):
        node = root
        # Selection
        while node.children:
            node = max(node.children, key=lambda n: n.ucb1(root.visits))
        # Expansion
        if node.visits > 0:
            # Expand node with possible next states
            for child_state in get_possible_states(node.state):
                node.children.append(Node(child_state, node))
            if node.children:
                node = random.choice(node.children)
        # Simulation
        result = simulate_fn(node.state)
        # Backpropagation
        while node:
            node.visits += 1
            node.wins += result  # Assume result 1=win, 0=lose/draw
            node = node.parent
    # Choose best action:
    best_child = max(root.children, key=lambda n: n.visits)
    return best_child.state

def get_possible_states(state):
    # Game-specific logic to generate states
    pass

def simulate_fn(state):
    # Game-specific playout/simulation
    # Return 1 if win, 0 if loss/draw for player to move
    pass