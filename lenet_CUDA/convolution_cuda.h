#ifndef CONVOLUTION_CUDA_H
#define CONVOLUTION_CUDA_H

#include <iostream>
#include <vector>

// Function prototype for averagePooling.
extern "C"{
std::vector<std::vector<std::vector<float>>> convolve2dDeep_GPU(
    const std::vector<std::vector<std::vector<float>>> &inputMaps,
    const std::vector<std::vector<std::vector<std::vector<float>>>> &kernels,
    const std::vector<float> &biases,
    const int stride,
    const int padding);
}
#endif // CONVOLUTION_CUDA_H