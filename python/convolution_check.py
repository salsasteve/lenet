import scipy.signal as signal
import numpy as np

# Load the image from the binary file
x_file_path = "../mnist_data/mnist_x_test.bin"
x_test = np.fromfile(x_file_path, dtype=np.float32).reshape(-1, 28, 28)

# Select the first image
image = x_test[0]

# read binary kernels

kernels_file_path = "../read_model/parameters/conv2d_1_weights.bin"
kernels = np.fromfile(kernels_file_path, dtype=np.float32).reshape(-1, 5, 5)

# read binary biases

biases_file_path = "../read_model/parameters/conv2d_1_bias.bin"
biases = np.fromfile(biases_file_path, dtype=np.float32)

# Perform the convolution and add the bias
output = []
for i, kernel in enumerate(kernels):
    convolved = signal.convolve2d(image, kernel, mode='same')
    convolved_with_bias = convolved + biases[i]  # Add bias to each element
    output.append(convolved_with_bias)

# Print dimensions of the output and save it
for i, convolved_with_bias in enumerate(output):
    print("Output", i, convolved_with_bias.shape)

def save_to_binary(data, filename):
    # Convert data to float32 for a consistent data type, if it's image data
    if len(data.shape) > 1:  # More than one dimension indicates image data
        data = data.astype(np.float32)
    else:  # Label data, keep as integers
        data = data.astype(np.int32)
    # Write the data to a file
    data.tofile(filename)


# Save the output to a file
output_file_path = "../mnist_data/convolution_output.bin"

save_to_binary(np.array(output), output_file_path)
