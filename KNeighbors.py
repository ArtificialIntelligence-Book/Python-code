from sklearn.neighbors import KNeighborsClassifier

X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

print("k-NN prediction:", knn.predict([[1.1]]))