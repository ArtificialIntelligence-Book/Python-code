from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

X = [[0, 0], [1, 1], [0.5, 0.5]]
y = [0, 1, 0]

bdt = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=10,
    random_state=42)
bdt.fit(X, y)

print("AdaBoost prediction:", bdt.predict([[0.7, 0.7]]))