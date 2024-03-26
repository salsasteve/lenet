import tensorflow as tf
import numpy as np
import os

# Define the file paths for convenience
x_file_path = "../mnist_data/mnist_x_test.bin"
y_file_path = "../mnist_data/mnist_y_test.bin"

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(_, _), (x_test, y_test) = mnist.load_data()

# Function to save a numpy array to a binary file
def save_to_binary(data, filename):
    # Convert data to float32 for a consistent data type, if it's image data
    if len(data.shape) > 1:  # More than one dimension indicates image data
        data = data.astype(np.float32)
    else:  # Label data, keep as integers
        data = data.astype(np.int32)
    # Write the data to a file
    data.tofile(filename)


# print first element of x_test
print(x_test[0])

# print first element of y_test
print(y_test[0])


# normalize the data

x_test = x_test / 255.0
y_test = y_test / 255.0


# print first element of x_test
print(x_test[0])

# print first element of y_test
print(y_test[0])

# print dimensions of the data
print("x_test shape", x_test.shape)
print("y_test shape", y_test.shape)


# Check if either file does not exist and then proceed to save
if not all(os.path.exists(path) for path in [x_file_path, y_file_path]):
    save_to_binary(x_test, x_file_path)
    save_to_binary(y_test, y_file_path)
else:
    print("Files already exist")

