import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X, _ = load_iris(return_X_y=True)

model = KMeans(n_clusters=3, n_init=10, random_state=42)
labels = model.fit_predict(X)

print("WCSS (Inertia):", model.inertia_)
print("Silhouette Score:", silhouette_score(X, labels))

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c='red', marker='X', s=200)
plt.title("K-Means Clustering"); plt.savefig("plot.png")