import numpy as np

def sigmoid3D(v):
    return 1.0 / (1.0 + np.exp(-v))

def tanh3D(v):
    return np.tanh(v)

def softmax3D(v):
    exp_v = np.exp(v - np.max(v, axis=-1, keepdims=True))  # For numerical stability
    return exp_v / np.sum(exp_v, axis=-1, keepdims=True)

def relu3D(v):
    return np.maximum(0.0, v)

def main():
    # Test data
    data = np.array([
        [[1.0, 2.0], [3.0, 4.0]], 
        [[5.0, 6.0], [7.0, 8.0]]
    ])

    # Sigmoid
    sigmoid_result = sigmoid3D(data)
    print("Sigmoid:\n", sigmoid_result, "\n")

    # Tanh
    tanh_result = tanh3D(data)
    print("Tanh:\n", tanh_result, "\n")

    # Softmax
    softmax_result = softmax3D(data)
    print("Softmax:\n", softmax_result, "\n")

    # ReLU
    relu_result = relu3D(data)
    print("ReLU:\n", relu_result, "\n")

if __name__ == "__main__":
    main()
