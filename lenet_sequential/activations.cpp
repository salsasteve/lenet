#include "activations.h"
#include <iostream>
#include <vector>
#include <cmath> // For exp and tanh functions

// Sigmoid function applied to a 3D vector
std::vector<std::vector<std::vector<float>>> sigmoid3D(const std::vector<std::vector<std::vector<float>>>& v) {
    std::vector<std::vector<std::vector<float>>> output;
    for (const auto& matrix : v) {
        std::vector<std::vector<float>> new_matrix;
        for (const auto& row : matrix) {
            std::vector<float> new_row;
            for (auto val : row) {
                new_row.push_back(1.0f / (1.0f + std::exp(-val)));
            }
            new_matrix.push_back(new_row);
        }
        output.push_back(new_matrix);
    }
    return output;
}

// Hyperbolic tangent function applied to a 3D vector
std::vector<std::vector<std::vector<float>>> tanh3D(const std::vector<std::vector<std::vector<float>>>& v) {
    std::vector<std::vector<std::vector<float>>> output;
    for (const auto& matrix : v) {
        std::vector<std::vector<float>> new_matrix;
        for (const auto& row : matrix) {
            std::vector<float> new_row;
            for (auto val : row) {
                new_row.push_back(std::tanh(val));
            }
            new_matrix.push_back(new_row);
        }
        output.push_back(new_matrix);
    }
    return output;
}

// Hyperbolic tangent function applied to a 1D vector
std::vector<float> tanh1D(const std::vector<float>& v) {
    std::vector<float> output;
    for (auto val : v) {
        output.push_back(std::tanh(val));
    }
    return output;
}

// Softmax function applied to each row of each 2D matrix in a 3D vector
std::vector<std::vector<std::vector<float>>> softmax3D(const std::vector<std::vector<std::vector<float>>>& v) {
    std::vector<std::vector<std::vector<float>>> output;
    for (const auto& matrix : v) {
        std::vector<std::vector<float>> new_matrix;
        for (const auto& row : matrix) {
            float sum = 0.0f;
            std::vector<float> new_row(row.size());

            // Compute the sum of exp(values) for each row
            for (auto val : row) {
                sum += std::exp(val);
            }

            // Compute softmax for each value in the row
            for (size_t i = 0; i < row.size(); ++i) {
                new_row[i] = std::exp(row[i]) / sum;
            }
            new_matrix.push_back(new_row);
        }
        output.push_back(new_matrix);
    }
    return output;
}

// Softmax function applied to a 1D vector
std::vector<float> softmax1D(const std::vector<float>& v) {
    float sum = 0.0f;
    std::vector<float> output(v.size());

    // Compute the sum of exp(values) for the vector
    for (auto val : v) {
        sum += std::exp(val);
    }

    // Compute softmax for each value in the vector
    for (size_t i = 0; i < v.size(); ++i) {
        output[i] = std::exp(v[i]) / sum;
    }
    return output;
}

// ReLU function applied to a 3D vector
std::vector<std::vector<std::vector<float>>> relu3D(const std::vector<std::vector<std::vector<float>>>& v) {
    std::vector<std::vector<std::vector<float>>> output;
    for (const auto& matrix : v) {
        std::vector<std::vector<float>> new_matrix;
        for (const auto& row : matrix) {
            std::vector<float> new_row;
            for (auto val : row) {
                new_row.push_back(std::max(0.0f, val));
            }
            new_matrix.push_back(new_row);
        }
        output.push_back(new_matrix);
    }
    return output;
}

// Tested against python in the python folder
// Tested against the following main function:
// int main() {
//     // Test data
//     std::vector<std::vector<std::vector<float>>> data = {
//         {{1.0f, 2.0f}, {3.0f, 4.0f}}, 
//         {{5.0f, 6.0f}, {7.0f, 8.0f}}
//     };

//     // Sigmoid
//     std::vector<std::vector<std::vector<float>>> sigmoid_result = sigmoid3D(data);
//     std::cout << "Sigmoid:\n";
//     for (const auto& matrix : sigmoid_result) {
//         for (const auto& row : matrix) {
//             for (auto val : row) std::cout << val << ' ';
//             std::cout << '\n';
//         }
//         std::cout << "---\n";
//     }

//     // Tanh
//     std::vector<std::vector<std::vector<float>>> tanh_result = tanh3D(data);
//     std::cout << "Tanh:\n";
//     for (const auto& matrix : tanh_result) {
//         for (const auto& row : matrix) {
//             for (auto val : row) std::cout << val << ' ';
//             std::cout << '\n';
//         }
//         std::cout << "---\n";
//     }

//     // Softmax
//     std::vector<std::vector<std::vector<float>>> softmax_result = softmax3D(data);
//     std::cout << "Softmax:\n";
//     for (const auto& matrix : softmax_result) {
//         for (const auto& row : matrix
//         ) {
//             for (auto val : row) std::cout << val << ' ';
//             std::cout << '\n';
//         }
//         std::cout << "---\n";
//     }

//     // ReLU
//     std::vector<std::vector<std::vector<float>>> relu_result = relu3D(data);
//     std::cout << "ReLU:\n";
//     for (const auto& matrix : relu_result) {
//         for (const auto& row : matrix) {
//             for (auto val : row) std::cout << val << ' ';
//             std::cout << '\n';
//         }
//         std::cout << "---\n";
//     }

//     return 0;
// }