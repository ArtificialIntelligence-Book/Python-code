from sklearn.mixture import GaussianMixture

X = np.vstack((
    np.random.randn(100, 2) + np.array([0, 0]),
    np.random.randn(100, 2) + np.array([5, 5])
))

gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
gmm.fit(X)

print("GMM means:\n", gmm.means_)
print("GMM prediction for a sample:", gmm.predict([[1, 1]]))