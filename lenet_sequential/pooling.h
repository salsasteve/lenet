#ifndef POOLING_H
#define POOLING_H

#include <iostream>
#include <vector>

// Function prototype for averagePooling.
std::vector<std::vector<float>> averagePooling(const std::vector<std::vector<float>>& input,
                                               int poolSize,
                                               int stride);

std::vector<std::vector<std::vector<float>>> averagePooling3D(const std::vector<std::vector<std::vector<float>>>& input,
                                                               int poolSize,
                                                               int stride);

#endif // POOLING_H