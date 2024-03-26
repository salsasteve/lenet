#ifndef CONVOLUTION_LAYER_H
#define CONVOLUTION_LAYER_H

#include <vector>

class ConvolutionLayer {
public:
    // Constructor declaration
    ConvolutionLayer(
        size_t inputHeight, size_t inputWidth, size_t inputDepth,
        size_t filterHeight, size_t filterWidth,
        size_t horizontalStride, size_t verticalStride,
        size_t paddingHeight, size_t paddingWidth,
        size_t numFilters,
        const std::vector<std::vector<std::vector<std::vector<float>>>>& initialFilters = {},
        bool useInitialFilters = false);

    // Method declarations
    void Forward(const std::vector<std::vector<std::vector<float>>>& input,
                 std::vector<std::vector<std::vector<float>>>& output);

    std::vector<std::vector<std::vector<std::vector<float>>>> getFilters() const;

private:
    // Attributes
    size_t inputHeight, inputWidth, inputDepth;
    size_t filterHeight, filterWidth;
    size_t horizontalStride, verticalStride;
    size_t paddingHeight, paddingWidth;
    size_t numFilters;
    std::vector<std::vector<std::vector<std::vector<float>>>> filters;

    // Private method declarations
    void initializeFiltersRandomly();
    void initializeFilters(const std::vector<std::vector<std::vector<std::vector<float>>>>& initialFilters);
};

#endif // CONVOLUTION_LAYER_H
