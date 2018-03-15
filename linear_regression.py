import numpy as np
from sklearn.linear_model import LinearRegression

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Pizza diameter
# X should be 2D array
X_train = np.array([6, 8, 10, 14, 18]).reshape(-1, 1)

# Pizza prices
y_train = [7, 9, 13, 17.5, 18]


plt.figure()
plt.title('Pizaa price plotted against diameter')
plt.xlabel('Diameter in inches')
plt.ylabel('Price in dollars')
plt.plot(X_train, y_train, 'ko')
plt.axis([0, 25, 0, 25])
plt.grid(True)
plt.plot([0, 25], [1.97, 26.37], color='k', linestyle='-', linewidth=2)
# plt.show()

model = LinearRegression()

model.fit(X_train, y_train)

test_pizza = np.array([[12]])
predicted_price = model.predict(test_pizza)[0]

print('A 12 pizza should cost $%.2f' % predicted_price)

# RSS Residual sum fo squares
rss = np.mean((model.predict(X_train) - y_train)**2)
print('Residual sum fo squares: %.2f' % rss)


# Evaluating the model
X_test = np.array([8, 9, 11, 16, 12]).reshape(-1, 1)
y_test = [11, 8.5, 15, 18, 11]
r_squared = model.score(X_test, y_test)
print(r_squared)