#ifndef DENSE_LAYER_CUDA_H
#define DENSE_LAYER_CUDA_H

#include <iostream>
#include <vector>
extern "C"{
std::vector<float> dense_GPU(
    std::vector<float> &input,
    std::vector<float> &bias,
    std::vector<std::vector<float>> &weights,
    int numOutputs)
}
#endif // DENSE_LAYER_CUDA_H