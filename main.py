import numpy as np
import matrixann as mann


firstLayer = mann.Layer(
    np.array([[3, 4], [6, 5]], dtype=float), np.array([1, -6], dtype=float))
secondLayer = mann.Layer(np.array([[2, 4]], dtype=float),
                         np.array([-3.92], dtype=float))

network = mann.Network(layers=[firstLayer, secondLayer], learning_rate=0.01)

input = np.array([1, 0])
output = np.array([1], dtype=float)

network.forwardPass(input)
network.backwardPass(input, output)
