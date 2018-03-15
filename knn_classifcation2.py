import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_score

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

from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier

# Conver label to 0, 1
lb = LabelBinarizer()
y_train_binarized = lb.fit_transform(y_train)
K = 3
clf = KNeighborsClassifier(n_neighbors= K)
clf.fit(X_train, y_train_binarized.reshape(-1))

prediction_binarized = clf.predict(np.array([155, 70]).reshape(1, -1))[0]
predicted_label = lb.inverse_transform(prediction_binarized)
print(predicted_label)


X_test = np.array([
    [168, 65],
    [180, 96],
    [160, 52],
    [169, 67]
])

y_test = ['male', 'male','female', 'female']
y_test_binarized = lb.transform(y_test)
score = precision_score(y_test_binarized, prediction_binarized)