#include "convolution_layer.h"
#include <random>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <iostream>

ConvolutionLayer::ConvolutionLayer(
    size_t inputHeight, size_t inputWidth, size_t inputDepth,
    size_t filterHeight, size_t filterWidth,
    size_t horizontalStride, size_t verticalStride,
    size_t paddingHeight, size_t paddingWidth,
    size_t numFilters)
    : inputHeight(inputHeight), inputWidth(inputWidth), inputDepth(inputDepth),
      filterHeight(filterHeight), filterWidth(filterWidth),
      horizontalStride(horizontalStride), verticalStride(verticalStride),
      paddingHeight(paddingHeight), paddingWidth(paddingWidth),
      numFilters(numFilters) {
    initializeFilters();
}

void ConvolutionLayer::initializeFilters() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1); // Standard normal distribution

    filters.resize(numFilters);
    for (auto& filter : filters) {
        filter.resize(inputDepth);
        for (auto& slice : filter) {
            slice.resize(filterHeight, std::vector<float>(filterWidth, 0.0));
            for (auto& row : slice) {
                for (float& weight : row) {
                    do {
                        weight = d(gen); // Sample from normal distribution
                    } while (std::abs(weight) > 2.0); // Truncate at 2 standard deviations
                }
            }
        }
    }
}


void ConvolutionLayer::Forward(const std::vector<std::vector<std::vector<float>>>& input,
                               std::vector<std::vector<std::vector<float>>>& output) {
                    
    // Create a padded input
    size_t paddedHeight = inputHeight + 2 * paddingHeight;
    size_t paddedWidth = inputWidth + 2 * paddingWidth;
    std::vector<std::vector<std::vector<float>>> paddedInput(inputDepth,
        std::vector<std::vector<float>>(paddedHeight, std::vector<float>(paddedWidth, 0.0)));


    // Copy the original input into the center of the padded input
    for (size_t k = 0; k < inputDepth; ++k) {
        for (size_t i = 0; i < inputHeight; ++i) {
            for (size_t j = 0; j < inputWidth; ++j) {
                paddedInput[k][i + paddingHeight][j + paddingWidth] = input[k][i][j];
            }
        }
    }

    // Perform the convolution on the padded input
    for (size_t f = 0; f < numFilters; ++f) {
        for (size_t i = 0; i <= paddedHeight - filterHeight; i += verticalStride) {
            for (size_t j = 0; j <= paddedWidth - filterWidth; j += horizontalStride) {
                float sum = 0.0;
                for (size_t k = 0; k < inputDepth; ++k) {
                    for (size_t di = 0; di < filterHeight; ++di) {
                        for (size_t dj = 0; dj < filterWidth; ++dj) {
                            sum += paddedInput[k][i + di][j + dj] * filters[f][k][di][dj];
                        }
                    }
                }
                output[f][i / verticalStride][j / horizontalStride] = sum;
            }
        }
    }
}


std::vector<std::vector<std::vector<std::vector<float>>>> ConvolutionLayer::getFilters() const {
    return filters;
}
