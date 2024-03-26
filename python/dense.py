import numpy as np

class SimpleDenseLayer:
    def __init__(self, input_size, output_size):
        # Initialize weights and biases randomly
        self.weights = intital()  [.1,.3] []
        self.biases = np.random.randn(output_size, 1)
    
    def forward(self, input):
        # Compute the output of the dense layer
        return np.dot(self.weights, input) + self.biases

# Example usage
input_size = 5
output_size = 3
input_vector = np.random.randn(input_size, 1)  # An example input vector

# Create a SimpleDenseLayer instance
dense_layer = SimpleDenseLayer(input_size, output_size)

# Perform a forward pass through the layer
output = dense_layer.forward(input_vector)
print("Output of the dense layer:", output)
