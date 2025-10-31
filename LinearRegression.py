import numpy as np

class LinearRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None
    
    def fit(self, X, y):
        """
        Fit linear regression model using the Normal Equation
        X: numpy array shape (n_samples, n_features)
        y: numpy array (n_samples,)
        """
        # Add bias term (constant 1) column to X
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # Normal equation: theta = (X_b.T X_b)^(-1) X_b.T y
        theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.intercept = theta[0]
        self.coefficients = theta[1:]
    
    def predict(self, X):
        return np.dot(X, self.coefficients) + self.intercept


# Example usage:
X = np.array([[1], [2], [3], [4], [5]])  # feature
y = np.array([3, 4, 2, 5, 6])            # target

lr = LinearRegression()
lr.fit(X, y)
print("Linear Regression predictions:", lr.predict(X))