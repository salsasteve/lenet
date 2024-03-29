import numpy as np
from scipy.signal import convolve2d, correlate2d

test_images = [
    [  # Kernel 1
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]
    ],
    [  # Kernel 2
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0]
    ],
    [  # Kernel 3
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]
    ],
    [  # Kernel 4
        [-1, -1, -1, -1, -1],
        [-1, 2, 2, 2, -1],
        [-1, 2, 8, 2, -1],
        [-1, 2, 2, 2, -1],
        [-1, -1, -1, -1, -1]
    ],
    [  # Kernel 5
        [0, -1, 0, -1, 0],
        [-1, 0, -1, 0, -1],
        [0, -1, 4, -1, 0],
        [-1, 0, -1, 0, -1],
        [0, -1, 0, -1, 0]
    ],
    [  # Kernel 6
        [-2, -1, 0, -1, -2],
        [-1, 0, 1, 0, -1],
        [0, 1, 2, 1, 0],
        [-1, 0, 1, 0, -1],
        [-2, -1, 0, -1, -2]
    ]
]

# # Pad test images
# for i in range(len(test_images)):
#     test_images[i] = np.pad(test_images[i], ((2, 2), (2, 2)), 'constant', constant_values=(0, 0))


# print(test_images)



# Kernel
kernel = np.ones((5, 5), dtype=int)

# Convolution

for i in range(len(test_images)):
    output_matrix_scipy = convolve2d(test_images[i], kernel, mode='same')
    print(output_matrix_scipy+1)

for i in range(len(test_images)):
    output_matrix_scipy = correlate2d(test_images[i], kernel, mode='same')
    print(output_matrix_scipy+1)