import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.DataFrame({
    'Milk':  [1, 0, 1, 1, 1],
    'Bread': [1, 1, 1, 1, 0],
    'Eggs':  [0, 1, 1, 0, 1]
}).astype(bool)

freq = apriori(df, min_support=0.4, use_colnames=True)
rules = association_rules(freq, metric="confidence", min_threshold=0.6)
print("Frequent Itemsets:\n", freq)
print("\nAssociation Rules:\n", rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

s = pd.Series([10, 12, 14, 15, 13, 100, -50])
q1, q3 = s.quantile(0.25), s.quantile(0.75)
iqr = q3 - q1
outliers = s[(s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)]
print("\nOutliers:\n", outliers)