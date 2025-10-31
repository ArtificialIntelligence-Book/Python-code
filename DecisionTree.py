from sklearn.tree import DecisionTreeClassifier

X = [[0, 0], [1, 1]]
y = [0, 1]

dtc = DecisionTreeClassifier()
dtc.fit(X, y)

print("Decision Tree prediction:", dtc.predict([[2, 2]]))