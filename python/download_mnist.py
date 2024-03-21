import tensorflow as tf
import numpy as np
import os

# Define the file paths for convenience
x_file_path = '../mnist_data/mnist_x_merged.bin'
y_file_path = '../mnist_data/mnist_y_merged.bin'

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Merge the training and testing datasets
x_merged = np.concatenate((x_train, x_test), axis=0)
y_merged = np.concatenate((y_train, y_test), axis=0)

# Function to save a numpy array to a binary file
def save_to_binary(data, filename):
    # Convert data to float32 for a consistent data type, if it's image data
    if len(data.shape) > 1:  # More than one dimension indicates image data
        data = data.astype(np.float32)
    else:  # Label data, keep as integers
        data = data.astype(np.int32)
    # Write the data to a file
    data.tofile(filename)

# print first element of x_merged
print(x_merged[0])

# print first element of y_merged
print(y_merged[0])

# Check if either file does not exist and then proceed to save
if not all(os.path.exists(path) for path in [x_file_path, y_file_path]):
    save_to_binary(x_merged, x_file_path)
    save_to_binary(y_merged, y_file_path)
else:
    print('Files already exist')