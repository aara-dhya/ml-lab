from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

X = load_iris().data
m = KMeans(n_clusters=3, random_state=0, n_init=10).fit(X)

print("Clusters:", m.labels_)
print("Inertia:", m.inertia_)