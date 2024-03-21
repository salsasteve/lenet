from scipy import signal
import numpy as np


# Redefine 3x3 input and kernel for demonstration
input_3x3 = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

kernel_3x3 = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

# Perform convolution using 'same' mode to keep the output size equal to the input size
output_3x3 = signal.convolve2d(input_3x3, kernel_3x3, boundary='fill', mode='same')

print(output_3x3)
