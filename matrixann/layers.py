import numpy as np
from activation_functions import sigmoid


class Layer():
    # Weights is n*m matrix, row n being the weights for node n in the layer, and m being the weights for each input
    def __init__(self, weights: np.ndarray, biases: np.ndarray):
        self.weights: np.ndarray = weights
        self.biases: np.ndarray = biases
        self.output: np.ndarray | None = None

    # The calculation for a forward pass is (weights * inputs) + biases.
    # For a layer of n nodes with m inputs, the output is a vector with dimensions n
    def forwardPass(self, inputs: np.ndarray) -> None:
        result = np.dot(self.weights, inputs) + self.biases
        self.output = sigmoid(result)
