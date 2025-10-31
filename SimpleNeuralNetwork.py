import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        # Initialize weights and biases randomly
        self.w1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros(output_size)
        self.lr = learning_rate
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, x):
        # Forward pass
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, x, y, output):
        # Calculate output error
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output)
        
        # Error propagated to hidden layer
        hidden_error = output_delta.dot(self.w2.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.a1)
        
        # Update weights and biases
        self.w2 += self.a1.T.dot(output_delta) * self.lr
        self.b2 += np.sum(output_delta, axis=0) * self.lr
        self.w1 += np.atleast_2d(x).T.dot(hidden_delta) * self.lr
        self.b1 += np.sum(hidden_delta, axis=0) * self.lr
    
    def train(self, x, y, epochs=10000):
        for _ in range(epochs):
            output = self.forward(x)
            self.backward(x, y, output)

# Example: XOR problem (not linearly separable)
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])
Y = np.array([
    [0],
    [1],
    [1],
    [0]
])

nn = SimpleNeuralNetwork(2, 2, 1, learning_rate=0.5)
nn.train(X, Y, epochs=10000)

# Testing
for x in X:
    print(f"Input: {x}, Output: {nn.forward(x).round(3)}")