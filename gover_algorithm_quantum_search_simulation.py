"""
Grover's Algorithm - Simplified Quantum Search Simulation in Python

Grover's algorithm provides a quadratic speedup for unstructured search problems
by amplifying the probability amplitude of the correct solution(s) using quantum
amplitude amplification.

This code simulates Grover's algorithm for a search problem using classical linear algebra
to illustrate the core quantum principles in a small state space.

Key points:
- State vector initialized in uniform superposition.
- Oracle flips phase of the target state.
- Diffusion operator amplifies the target amplitude.
- Iterative application increases probability of measuring the target.
- Demonstrates how Grover's algorithm can be a building block for AI optimization.

Potential AI applications:
- Searching large solution spaces (e.g., hyperparameter tuning, combinatorial optimization).
- Speeding up constraint satisfaction problems.
- Quantum-enhanced heuristic search or sampling.

---

Dependencies:
- numpy

Run this script to see the probability evolution of the target state with each Grover iteration.
"""

import numpy as np

def create_uniform_superposition(n_qubits):
    """
    Creates a uniform superposition state vector for n_qubits.
    Size = 2^n_qubits, each amplitude = 1/sqrt(N)
    """
    N = 2 ** n_qubits
    state = np.ones(N) / np.sqrt(N)
    return state

def oracle(state, target_index):
    """
    Oracle operator flips the phase of the target state.
    """
    new_state = state.copy()
    new_state[target_index] *= -1
    return new_state

def diffusion_operator(n_qubits):
    """
    Constructs the diffusion (inversion about the mean) operator matrix.
    D = 2|s><s| - I
    where |s> is uniform superposition
    """
    N = 2 ** n_qubits
    s = np.ones((N, N)) / N
    I = np.identity(N)
    D = 2 * s - I
    return D

def grover_iteration(state, target_index, D):
    """
    Performs one Grover iteration: Oracle + Diffusion
    """
    state = oracle(state, target_index)
    state = D @ state
    return state

def measure_probabilities(state):
    """
    Computes probabilities of measuring each state.
    """
    return np.abs(state) ** 2

def simulate_grover(n_qubits, target_index, num_iterations):
    """
    Simulates Grover's algorithm on n_qubits with given target.
    Prints probabilities of the target state after each iteration.
    """
    state = create_uniform_superposition(n_qubits)
    D = diffusion_operator(n_qubits)

    print(f"Initial uniform superposition probability of target state ({target_index}): {measure_probabilities(state)[target_index]:.4f}")

    for i in range(num_iterations):
        state = grover_iteration(state, target_index, D)
        probs = measure_probabilities(state)
        print(f"After iteration {i+1}: Probability of target state = {probs[target_index]:.4f}")

    return probs

if __name__ == "__main__":
    # Number of qubits (search space size = 2^n_qubits)
    n_qubits = 3  # 8 states total

    # Index of the target solution (0-based)
    target_index = 5

    # Number of Grover iterations (approx ~ pi/4 * sqrt(N))
    num_iterations = 2

    print("=== Grover's Algorithm Simulation ===")
    final_probs = simulate_grover(n_qubits, target_index, num_iterations)

    print("\nFinal probability distribution over states:")
    for i, p in enumerate(final_probs):
        print(f"State {i}: Probability {p:.4f}")

"""
Notes:

- This simulation uses numpy arrays to represent quantum states.
- The oracle flips the amplitude sign of the target index (phase inversion).
- The diffusion operator amplifies target amplitude by inverting about the mean.
- Grover's algorithm approximately needs ~π/4 * sqrt(N) iterations for max success probability.
- In AI, Grover's algorithm concepts inspire efficient search and optimization heuristics.
"""