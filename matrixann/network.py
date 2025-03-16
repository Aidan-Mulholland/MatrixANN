import numpy as np
from .activation_functions import sigmoid_derivative


class Network():
    def __init__(self, layers, learning_rate):
        self.layers = layers
        self.learning_rate = learning_rate

    # For each layer in the network, call the forwardPass function
    def forwardPass(self, inputs):
        # Defining a variable to store the outputs from each layers fowardpass function
        previous_output = inputs
        for layer in self.layers:
            layer.forwardPass(previous_output)
            # Updating output to store layer output which is then used for the next layer
            previous_output = layer.output
        # Output the result
        return previous_output

    def backwardPass(self, inputs, expected_output):
        num_layers = len(self.layers)

        # Compute delta for output layer (last layer)
        output_layer = self.layers[-1]
        output_error = expected_output - output_layer.output  # c_j - u_j
        output_layer.delta = output_error * \
            sigmoid_derivative(output_layer.output)

        # Compute delta for hidden layers (working backwards)
        for l in range(num_layers - 2, -1, -1):  # Skip output layer, go backwards
            layer = self.layers[l]
            next_layer = self.layers[l + 1]
            layer.delta = np.dot(
                next_layer.weights.T, next_layer.delta) * sigmoid_derivative(layer.output)

        # Update weights and biases
        for l in range(num_layers):
            layer = self.layers[l]
            # Input for first layer, previous layer output otherwise
            inputs = inputs if l == 0 else self.layers[l - 1].output
            layer.weights += self.learning_rate * np.outer(layer.delta, inputs)
            layer.biases += self.learning_rate * layer.delta

        return output_error
