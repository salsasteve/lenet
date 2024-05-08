#ifndef DENSE_LAYER_CUDA_H
#define DENSE_LAYER_CUDA_H

#include <iostream>
#include <vector>

std::vector<float> dense_GPU(
    std::vector<float> &input,
    std::vector<float> &biases,
    std::vector<std::vector<float>> &weights,
    int numOutputs,
    bool activate);

#endif // DENSE_LAYER_CUDA_H
