from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt

X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])

Z = linkage(X, 'ward')

# Plot dendrogram
plt.figure(figsize=(6, 4))
dendrogram(Z)
plt.title("Hierarchical Clustering Dendrogram")
plt.show()

# Cut tree into flat clusters
labels = fcluster(Z, t=3, criterion='maxclust')
print("Hierarchical clustering labels:", labels)