// Convolution.h
#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <vector>

// Declare the 2D convolution function
void convolve2D(const std::vector<std::vector<int>>& input, 
                std::vector<std::vector<int>>& output, 
                const std::vector<std::vector<int>>& kernel);

#endif // CONVOLUTION_H
