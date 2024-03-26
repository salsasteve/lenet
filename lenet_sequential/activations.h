#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <vector>
#include <cmath> // For exp, tanh, and max

// Sigmoid function applied to a 3D vector
std::vector<std::vector<std::vector<float>>> sigmoid3D(const std::vector<std::vector<std::vector<float>>>& v);

// Hyperbolic tangent function applied to a 3D vector
std::vector<std::vector<std::vector<float>>> tanh3D(const std::vector<std::vector<std::vector<float>>>& v);

// Softmax function applied to each row of each 2D matrix in a 3D vector
std::vector<std::vector<std::vector<float>>> softmax3D(const std::vector<std::vector<std::vector<float>>>& v);

// ReLU function applied to a 3D vector
std::vector<std::vector<std::vector<float>>> relu3D(const std::vector<std::vector<std::vector<float>>>& v);

#endif // ACTIVATIONS_H
