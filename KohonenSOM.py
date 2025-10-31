import numpy as np

class KohonenSOM:
    def __init__(self, m, n, dim, learning_rate=0.5, radius=None, decay_rate=0.05):
        """
        :param m, n: dimensions of the SOM grid
        :param dim: dimensionality of input vectors
        """
        self.m = m
        self.n = n
        self.dim = dim
        self.lr = learning_rate
        self.radius = radius if radius else max(m, n) / 2
        self.decay_rate = decay_rate
        self.weights = np.random.rand(m, n, dim)
    
    def _euclidean_distance(self, x, y):
        # Vector euclidean distance
        return np.linalg.norm(x - y)
    
    def _find_bmu(self, x):
        """
        Find best matching unit for input x
        """
        min_dist = float('inf')
        bmu_idx = (0, 0)
        for i in range(self.m):
            for j in range(self.n):
                dist = self._euclidean_distance(x, self.weights[i, j])
                if dist < min_dist:
                    min_dist = dist
                    bmu_idx = (i, j)
        return bmu_idx
    
    def _neighborhood_function(self, dist, radius):
        # Gaussian neighborhood function
        return np.exp(-dist**2 / (2 * (radius ** 2)))
    
    def train(self, data, num_iterations=1000):
        time_constant = num_iterations / np.log(self.radius)
        
        for t in range(num_iterations):
            # Decay learning rate and radius over time
            lr = self.lr * np.exp(-t / num_iterations)
            radius = self.radius * np.exp(-t / time_constant)
            
            # Pick a random input vector
            x = data[np.random.randint(0, data.shape[0])]
            
            # Find BMU
            bmu_i, bmu_j = self._find_bmu(x)
            
            # Update weights in neighborhood
            for i in range(self.m):
                for j in range(self.n):
                    # Distance on the grid
                    dist = np.sqrt((i - bmu_i) ** 2 + (j - bmu_j) ** 2)
                    if dist <= radius:
                        influence = self._neighborhood_function(dist, radius)
                        # Update weights towards input vector
                        self.weights[i, j] += influence * lr * (x - self.weights[i, j])
    
    def map_vector(self, x):
        """
        Maps input vector to BMU on grid
        """
        return self._find_bmu(x)


# Example data: 2D points clustered around two centers
data = np.vstack((
    np.random.randn(100, 2) * 0.5 + np.array([1, 1]),
    np.random.randn(100, 2) * 0.5 + np.array([5, 5])
))

som = KohonenSOM(10, 10, 2)
som.train(data, num_iterations=500)

# Map samples
for point in data[:5]:
    print(f"Point {point} mapped to node {som.map_vector(point)}")