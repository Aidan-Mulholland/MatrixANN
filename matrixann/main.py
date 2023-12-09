import numpy as np

# Takes a n x m matrix of weights, where n is the number of nodes and m is the number of weights and m is the number of inputs in to the layer and a 1xn vector for the biases
class Layer():
  def __init__(self, weights, biases):
    self.weights = weights
    self.biases = biases

  # The calculation for a forward pass is (inputs * weights) + biases, stores this value to the layer output
  def forwardPass(self, inputs):
    print("LAYER")
    print(inputs, "*", self.weights)
    print(np.dot(inputs, self.weights), "+", self.biases)
    self.output = np.add(np.dot(inputs, self.weights), self.biases)

class Network(): 
  def __init__(self, layers, inputs):
    self.layers = layers
    self.inputs = inputs

  # For each layer in the network, call the forwardPass function
  def forwardPass(self):
    # Defining a variable to store the outputs from each layers fowardpass function
    output = self.inputs
    for layer in self.layers:
      layer.forwardPass(output)
      # Updating output to store layer output which is then used for the next layer
      output = layer.output
    # Output the result
    print("OUTPUT", output)

firstLayer = Layer(np.matrix([[3, 6], [4, 5]]), np.matrix([1, -6]))
secondLayer = Layer(np.matrix([[2], [4]]), np.matrix([-3.92]))
network = Network([firstLayer, secondLayer], [1, 0])
network.forwardPass()