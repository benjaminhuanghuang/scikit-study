import numpy as np
from sklearn.linear_model import LinearRegression

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from collections import Counter

X_train = np.array([
    [158, 64],
    [170, 86],
    [183, 84],
    [191, 80],
    [155, 49],
    [163, 59],
    [180, 67],
    [158, 54],
    [170, 67],
])

y_train = ['male', 'male', 'male', 'male', 'female',
           'female', 'female', 'female', 'female']

plt.figure()
plt.title('Human Heights and Weights by Sex')
plt.xlabel('Height in cm')
plt.ylabel('Weight in kg')

for i, x in enumerate(X_train):
    plt.scatter(x[0], x[1], c='k', marker='x' if y_train[i] == 'male' else 'D')

plt.grid(True)
#plt.show()

x = np.array([[155, 70]])
distances = np.sqrt(np.sum((X_train - x)**2, axis=1))

nearest_neighbor_indices = distances.argsort()[:3]
nearest_neighbor_genders = np.take(y_train, nearest_neighbor_indices)
b = Counter(np.take(y_train, distances.argsort()[:3]))
sex = b.most_common(1)[0][0]
print(sex)