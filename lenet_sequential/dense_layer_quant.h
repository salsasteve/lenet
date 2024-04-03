// dense_layer.h
#ifndef DENSE_LAYER_QUANT_H
#define DENSE_LAYER_QUANT_H

#include <vector>
#include <cstdint>

// Function prototype for the dense layer.
std::vector<uint16_t> dense_quant(
    std::vector<float> &input,
    std::vector<uint16_t> &bias,
    std::vector<std::vector<uint16_t>> &weights,
    int numOutputs);

#endif // DENSE_LAYER_QUANT_H
