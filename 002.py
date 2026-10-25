from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = [10, 20, 30, 40, 50]

m = LinearRegression().fit(X, y)
print("Prediction:", m.predict([[6]]))
print("Accuracy:", m.score(X, y))