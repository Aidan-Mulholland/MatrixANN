import numpy as np
import matrixann as mann

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt


firstLayer = mann.Layer(
    np.array([[3, 4], [6, 5]], dtype=float), np.array([1, -6], dtype=float))
secondLayer = mann.Layer(np.array([[2, 4]], dtype=float),
                         np.array([-3.92], dtype=float))

network = mann.Network(layers=[firstLayer, secondLayer], learning_rate=0.01)

dataset = fetch_california_housing(as_frame=True)
print(dataset)

x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, test_size=0.2, random_state=42)

errors = []
for epoch in range(10):
    for row_index in range(len(x_train)):
        print(x_train.iloc[row_index, :2], x_train.iloc[row_index, -1])
        predicted = network.forwardPass(x_train.iloc[row_index, :2])
        error = network.backwardPass(
            x_train.iloc[row_index, :2], y_train.iloc[row_index])
    errors.append(error)

predicted = []
for row_index in range(len(x_test)):
    print(x_test.iloc[row_index, :2], x_test.iloc[row_index, -1])
    predict = network.forwardPass(x_test.iloc[row_index, :2])
    predicted.append(predict)

print(r2_score(predicted, y_test))


plt.plot(errors)
plt.show()
