#ifndef POOLING_CUDA_H
#define POOLING_CUDA_H

#include <iostream>
#include <vector>

// Function prototype for averagePooling.
extern "C"{
std::vector<std::vector<std::vector<float>>> averagePooling3D_GPU(
    const std::vector<std::vector<std::vector<float>>> &inputMaps,
    const int poolsize,
    const int stride);
}
#endif // POOLING_CUDA_H