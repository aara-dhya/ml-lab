from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)
m = LogisticRegression(max_iter=200).fit(X, y)

print("Prediction:", m.predict([X[0]]))
print("Accuracy:", m.score(X, y))