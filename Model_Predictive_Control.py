import numpy as np
from scipy.optimize import minimize

def mpc_step(current_state, goal, horizon=10):
    # Define cost function: distance to goal + control effort
    def cost(actions):
        state = current_state.copy()
        total_cost = 0
        for action in actions.reshape(-1, 2):
            state += action  # Simple dynamics
            total_cost += np.linalg.norm(state - goal) + 0.1 * np.linalg.norm(action)
        return total_cost
    
    # Optimize actions over horizon
    initial_guess = np.zeros(horizon * 2)
    result = minimize(cost, initial_guess, method='SLSQP')
    
    # Return only first action
    optimal_actions = result.x.reshape(-1, 2)
    return optimal_actions[0]

# Example usage
current_state = np.array([0.0, 0.0])
goal = np.array([10.0, 10.0])
action = mpc_step(current_state, goal)
new_state = current_state + action