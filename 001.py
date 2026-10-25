import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.DataFrame({
    'Age': [20, 30, np.nan, 40, 20],
    'Sal': [2000, 3000, 4000, np.nan, 2000],
    'City': ['A', 'B', 'A', 'B', 'A'],
    'Buy': ['No', 'Yes', 'No', 'Yes', 'No']
})

df = df.drop_duplicates()
df[['Age', 'Sal']] = df[['Age', 'Sal']].fillna(df[['Age', 'Sal']].mean())
df['Log_Sal'] = np.log(df['Sal'])
df['Buy'] = LabelEncoder().fit_transform(df['Buy'])
df = pd.get_dummies(df, columns=['City'])
df[['Age', 'Sal']] = StandardScaler().fit_transform(df[['Age', 'Sal']])
print("Processed Data:\n", df)

X, y = df.drop('Buy', axis=1), df['Buy']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
print("Train & Test Shapes:", X_tr.shape, X_te.shape)