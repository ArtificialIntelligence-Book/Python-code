import numpy as np

class HopfieldNetwork:
    def __init__(self, patterns):
        """
        :param patterns: List or array of bipolar (-1, +1) patterns to store
        """
        self.patterns = patterns
        n = patterns.shape[1]
        self.n = n
        self.weights = np.zeros((n, n))
        self.train(patterns)
    
    def train(self, patterns):
        """
        Training with Hebbian learning rule (no self-connections)
        """
        for p in patterns:
            self.weights += np.outer(p, p)
        np.fill_diagonal(self.weights, 0)
        self.weights /= self.n
    
    def activation(self, x):
        """
        Sign activation function
        """
        return np.where(x >= 0, 1, -1)
    
    def recall(self, input_pattern, max_iterations=10):
        """
        Recall stored pattern using parallel update
        """
        x = input_pattern.copy()
        for i in range(max_iterations):
            # Calculate net input for all neurons in parallel
            net_input = np.dot(self.weights, x)
            x_new = self.activation(net_input)
            if np.array_equal(x, x_new):
                break  # converged
            x = x_new
        return x


# Example with two stored patterns
patterns = np.array([
    [1, -1, 1, -1],
    [-1, 1, -1, 1]
])

hopfield = HopfieldNetwork(patterns)
# Test with a noisy version of first pattern
noisy_pattern = np.array([1, -1, -1, -1])
recalled = hopfield.recall(noisy_pattern)
print("Hopfield Network recalled pattern:", recalled)