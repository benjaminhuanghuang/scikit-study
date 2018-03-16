'''
    Wine database: 
        https://archive.ics.uci.edu/ml/machine-learning-databases/wine/
'''

import pandas as pd
df = pd.read_csv('data/winequality-red.csv', sep=';')
print(df.describe())

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.scatter(df['alcohol'], df['quality'])
plt.xlabel('Alcohol')
plt.ylabel('Quality')
plt.title('Alcohol Against Quality')
plt.show()


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score

X = df[list(df.columns)[:-1]]
y = df['quality']

X_train, X_test, y_trian, y_test = train_test_split(X, y)
regressor = LinearRegression()
regressor.fit(X_train, y_trian)
y_predictions = regressor.predict(X_test)
score = regressor.score(X_test, y_test)
score2 = cross_val_score(regressor, X, y, cv=5)
