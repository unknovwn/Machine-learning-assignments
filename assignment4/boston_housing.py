import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold, cross_val_score

data = load_boston()
X = scale(data["data"])
y = data["target"]

cross_validator = KFold(n_splits=5, shuffle=True, random_state=42)
mean_accuracies = []

p_values = np.linspace(1, 10, 20)
for p in p_values:
    regressor = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski', p=p)
    accuracy = cross_val_score(estimator=regressor, X=X, y=y, scoring='neg_mean_squared_error')
    mean_accuracies.append(accuracy.mean())

max_accuracy_index = mean_accuracies.index(max(mean_accuracies))
print('Optimal power parameter for Minkowski metric: {}'.format(p_values[max_accuracy_index]))