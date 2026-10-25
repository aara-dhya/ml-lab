import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.ensemble import IsolationForest

df = pd.DataFrame({
    'Milk': [1, 1, 0, 1],
    'Bread': [1, 1, 1, 0],
    'Butter': [0, 1, 1, 1]
}).astype(bool)

f = apriori(df, min_support=0.5, use_colnames=True)
print(f)
print(association_rules(f, metric="confidence", min_threshold=0.5))

print(IsolationForest(random_state=0).fit_predict(df.astype(int)))