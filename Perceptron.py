class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        # Initialize weights randomly and bias zero
        self.weights = np.zeros(input_size)
        self.bias = 0
        self.lr = learning_rate
    
    def activation(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, x):
        linear_output = np.dot(x, self.weights) + self.bias
        return self.activation(linear_output)
    
    def train(self, X, y, epochs=10):
        for _ in range(epochs):
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                error = target - prediction
                self.weights += self.lr * error * xi
                self.bias += self.lr * error


# Example: OR logic function
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 1, 1, 1])

perceptron = Perceptron(input_size=2)
perceptron.train(X, y, epochs=10)

for x in X:
    print(f"Input: {x}, Predicted: {perceptron.predict(x)}")