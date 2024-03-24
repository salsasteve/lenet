#ifndef CONV_LAYER_HPP  // Include guard
#define CONV_LAYER_HPP

#include <vector>
#include <cstdlib> // For size_t

// ConvolutionLayer class declaration
class ConvolutionLayer {
public:
    // Constructor
    ConvolutionLayer(
        size_t inputHeight,
        size_t inputWidth,
        size_t inputDepth,
        size_t filterHeight,
        size_t filterWidth,
        size_t horizontalStride,
        size_t verticalStride,
        size_t paddingHeight, 
        size_t paddingWidth,
        size_t numFilters);

    // Forward pass
    void Forward(const std::vector<std::vector<std::vector<float>>>& input,
                 std::vector<std::vector<std::vector<float>>>& output);

    // Getters for debugging or further processing
    std::vector<std::vector<std::vector<std::vector<float>>>> getFilters() const;

private:
    // Layer configurations
    size_t inputHeight, inputWidth, inputDepth;
    size_t filterHeight, filterWidth;
    size_t horizontalStride, verticalStride;
    size_t paddingHeight, paddingWidth;
    size_t numFilters;

    // Filters
    std::vector<std::vector<std::vector<std::vector<float>>>> filters; // 4D Vector [numFilters][filterDepth][filterHeight][filterWidth]

    // Utility functions
    void initializeFilters();
};

#endif // CONV_LAYER_HPP
