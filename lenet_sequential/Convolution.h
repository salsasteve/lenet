// Convolution.h
#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <vector>

// Declare the 2D convolution function
void convolve2D(const std::vector<std::vector<float>>& input, 
                std::vector<std::vector<float>>& output, 
                const std::vector<std::vector<float>>& kernel);

#endif // CONVOLUTION_H
