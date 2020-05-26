import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('titanic.csv')
df.info() # Age column has missing data
df['Cabin'] = df['Cabin'].fillna('unknown')
df_without_na = df.dropna() # Drop all objects with missing age

X = df_without_na[['Pclass', 'Fare', 'Age', 'Sex']]
X.replace(['female', 'male'], [0, 1], inplace=True) # Replace string values by numbers for decision tree
y = df_without_na['Survived']

clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, y)
importances = clf.feature_importances_
print(pd.DataFrame(data=[X.columns, importances]).transpose()) # Most important features: Fare (0.3047), Sex (0.2983)