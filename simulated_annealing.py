import math
import random

def simulated_annealing(start, neighbors_fn, heuristic, initial_temp=1000, cooling_rate=0.95, max_iter=1000):
    """
    Simulated Annealing algorithm.
    
    :param start: Initial state
    :param neighbors_fn: Function that returns neighbors of a state
    :param heuristic: Function returning heuristic value (to minimize)
    :param initial_temp: Starting temperature
    :param cooling_rate: Cooling rate (0 < cooling_rate < 1)
    :param max_iter: Max iterations to run
    :return: Best found state
    """
    current = start
    temperature = initial_temp
    best = current

    for i in range(max_iter):
        if temperature <= 0:
            break

        neighbors = neighbors_fn(current)
        if not neighbors:
            break

        next_state = random.choice(neighbors)
        delta_e = heuristic(current) - heuristic(next_state)

        # Accept better states or accept worse probabilistically
        if delta_e > 0 or math.exp(delta_e / temperature) > random.random():
            current = next_state
            if heuristic(current) < heuristic(best):
                best = current
        
        # Cool down temperature
        temperature *= cooling_rate

    return best


# Using same example as hill climbing
result = simulated_annealing(0, neighbors_fn, heuristic)
print("Simulated Annealing result:", result)