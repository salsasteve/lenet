// dense_layer.h
#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include <vector>

// Function prototype for the dense layer.
std::vector<float> dense(
    std::vector<float> &input,
    std::vector<float> &bias,
    std::vector<std::vector<float>> &weights,
    int numOutputs);

#endif // DENSE_LAYER_H
