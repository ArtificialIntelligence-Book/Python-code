from sklearn import svm

X = [[0, 0], [1, 1]]
y = [0, 1]

clf = svm.SVC(kernel='linear')
clf.fit(X, y)

print("SVM prediction:", clf.predict([[0.5, 0.5]]))