// dense_layer.h
#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include <vector>
#include <cstdint>

// Function prototype for the dense layer.
std::vector<float> dense(
    std::vector<float> &input,
    const std::vector<float> &bias,
    const std::vector<std::vector<float>> &weights,
    int numOutputs);

std::vector<float> dense_quantized(
    std::vector<float> &input,
    const std::vector<uint16_t> &bias,
    const std::vector<std::vector<uint16_t>> &weights,
    int numOutputs,
    const float weights_scale,
    const int weights_zero_points,
    const float biases_scale,
    const int biases_zero_points
);
#endif // DENSE_LAYER_H
