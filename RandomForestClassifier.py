from sklearn.ensemble import RandomForestClassifier

X = [[1, 2], [3, 4], [5, 6], [7, 8]]
y = [0, 0, 1, 1]

rfc = RandomForestClassifier(n_estimators=10, random_state=42)
rfc.fit(X, y)

print("Random Forest prediction:", rfc.predict([[4, 5]]))