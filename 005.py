import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(max_depth=3).fit(X_tr, y_tr)
pred = model.predict(X_te)

print("Accuracy:", accuracy_score(y_te, pred))
print("Confusion Matrix:\n", confusion_matrix(y_te, pred))
print("Report:\n", classification_report(y_te, pred))

plot_tree(model, filled=True); plt.savefig("plot.png")