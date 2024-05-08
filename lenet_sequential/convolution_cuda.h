#ifndef CONVOLUTION_CUDA_H
#define CONVOLUTION_CUDA_H

#include <iostream>
#include <vector>

// Function prototype for averagePooling.
vector<vector<vector<float>>> convolve2dDeep_GPU(
    const vector<vector<vector<float>>> &inputMaps,
    const vector<vector<vector<vector<float>>>> &kernels,
    const vector<float> &biases,
    const int stride,
    const int padding);

#endif // CONVOLUTION_CUDA_H
