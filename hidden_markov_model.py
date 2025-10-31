import numpy as np
from hmmlearn import hmm

# Define a Gaussian HMM with 2 states
model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=100)

# Simulated data: 100 samples with 1 feature
X = np.concatenate([
    np.random.normal(0, 1, (50, 1)),
    np.random.normal(5, 1, (50, 1))
])

model.fit(X)

hidden_states = model.predict(X)
print("Hidden states:\n", hidden_states)