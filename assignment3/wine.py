import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

names = [
    'class',
    'alcohol',
    'malic acid',
    'ash',
    'ash_alcalinity',
    'magnesium', 
    'total_phenols',
    'flavanoids',
    'nonflavanoid phenols',
    'proanthocyanins',
    'color_intensity',
    'hue',
    'od',
    'proline'
]
df = pd.read_csv(filepath_or_buffer='assignment3/wine.csv', names=names)
df.info() # No missing data

X = df.iloc[:, 1:] # Features
X_scaled = scale(X) # K neighbors method works bad without scaling
y = df['class']

cross_validator = KFold(n_splits=5, shuffle=True, random_state=42)
mean_accuracies = []
mean_accuracies_scaled_features = []

# Calculating optimal k without scaling
for k in range(1, 51):
    clf = KNeighborsClassifier(n_neighbors=k)
    accuracy = cross_val_score(estimator=clf, X=X, y=y, cv=cross_validator, scoring='accuracy')
    mean_accuracies.append(accuracy.mean())

max_accuracy = max(mean_accuracies)
optimal_k = mean_accuracies.index(max_accuracy) + 1
print('Without scaling:')
print('Optimal k: {}, accuracy: {}'.format(optimal_k, max_accuracy))
print()

# Calculating optimal k with scaled features
for k in range(1, 51):
    clf = KNeighborsClassifier(n_neighbors=k)
    accuracy = cross_val_score(estimator=clf, X=X_scaled, y=y, cv=cross_validator, scoring='accuracy')
    mean_accuracies_scaled_features.append(accuracy.mean())

max_accuracy_scaled_features = max(mean_accuracies_scaled_features)
optimal_k_scaled_features = mean_accuracies_scaled_features.index(max_accuracy_scaled_features) + 1
print('With scaling:')
print('Optimal k: {}, accuracy: {}'.format(optimal_k_scaled_features, max_accuracy_scaled_features))