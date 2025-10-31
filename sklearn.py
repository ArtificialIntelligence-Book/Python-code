from sklearn.linear_model import LinearRegression as SklearnLR

model = SklearnLR()
model.fit(X, y)
print("Sklearn Linear Regression predictions:", model.predict(X))