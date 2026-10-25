import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X, y = load_diabetes(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression().fit(X_tr, y_tr)
pred = model.predict(X_te)

print("MAE:", mean_absolute_error(y_te, pred))
print("MSE:", mean_squared_error(y_te, pred))
print("R2 Score:", r2_score(y_te, pred))

plt.scatter(y_te, pred)
plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.show()