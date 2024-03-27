#ifndef DENSE_LAYER_HPP
#define DENSE_LAYER_HPP

#include <vector>
#include <cmath>
#include <cassert>
#include <random>

#define DEBUG false
#define DEBUG_PREFIX "[DEBUG DENSE LAYER ]\t"


// using Input = std::vector<std::vector<float>>;
//using Output = std::vector<float>;

// params: input, weights, biases, activation
// the weights and biases have to be read in so may come as string
std::vector<float> forward
(
    std::vector<std::vector<std::vector<float>>>& input, // Input
    std::vector<float>& bias, // possibly a file name that has to be read
    std::vector<std::vector<float>>& weights, // possibly a file name that has to be read
    int numOutputs
)
{
    // first establish the size of the output matrix ???
    std::vector<float> output;
    output.resize(numOutputs);

    // flatten the matrix into a vector
    std::vector<float> intermediate_;
    for (size_t r = 0; r < input.size(); ++r) 
    {
        for (size_t c = 0; c < input[r].size(); ++c)
        {
            for (size_t d = 0; d < input[r][c].size(); ++d) {
                intermediate_.push_back(input[r][c][d]);
            }
        }
    }

    size_t height = input.size(), width = input[0].size(), depth = input[0][0].size();
    size_t flattenedSize = height * width * depth; // alternatively, intermediate_.size();

    // matrix multiplication
    for (size_t o = 0; o < numOutputs; ++o)
    {
        float sum = 0.0;
        size_t weightIdx = o * flattenedSize;
        
        for (size_t idx = 0; idx < flattenedSize; ++idx)
        {
            sum += weights[o][idx] * intermediate_[idx];
        }
        // include the bias
        sum += bias[o]; // works since the bias should theroetically be the same size as output for vector addition
        output[o] = sum;
    }

    // possibly include the activation before returning
    return output;

}

#endif

/*
for (int y = 0; y < outputHeight; ++y) {
        for (int x = 0; x < outputWidth; ++x) {
            float sum = 0.0;
            for (int m = 0; m < paddedInputMaps.size(); ++m) {
                for (int i = 0; i < kernelHeight; ++i) {
                    for (int j = 0; j < kernelWidth; ++j) {
                        sum += kernels[m][i][j] * paddedInputMaps[m][y * stride + i][x * stride + j];
                    }
                }
            }
            outputMap[y][x] = sum + bias; // Add the bias
        }
    }
*/