#ifndef DENSE_LAYER_CPP
#define DENSE_LAYER_CPP

#include "dense_layer.h"
#include <vector>
#include <cstdint>

std::vector<float> dense(
    std::vector<float> &input,
    const std::vector<float> &bias,
    const std::vector<std::vector<float>> &weights,
    int numOutputs)
{
    std::vector<float> output;
    output.resize(numOutputs);

    size_t flattenedSize = input.size();

    for (size_t o = 0; o < numOutputs; ++o)
    {
        float sum = 0.0;
        
        for (size_t idx = 0; idx < flattenedSize; ++idx)
        {
            sum += weights[idx][o] * input[idx];
        }

        sum += bias[o];
        output[o] = sum;
    }

    return output;
}

std::vector<float> dense_quantized(
    std::vector<float> &input,
    const std::vector<uint16_t> &bias,
    const std::vector<std::vector<uint16_t>> &weights,
    int numOutputs,
    const float weights_scale,
    const int weights_zero_points,
    const float biases_scale,
    const int biases_zero_points){
    std::vector<float> output;
    output.resize(numOutputs);

    size_t flattenedSize = input.size();

    for (size_t o = 0; o < numOutputs; ++o)
    {
        float sum = 0.0;
        int tmp = 0;
        for (size_t idx = 0; idx < flattenedSize; ++idx)
        {
            tmp += (static_cast<int>(weights[idx][o])-weights_zero_points) * static_cast<int>(input[idx]*255.);
        }

        sum += (1/255.) * weights_scale * static_cast<float>(tmp) + biases_scale * static_cast<float>(bias[o]-biases_zero_points);
        output[o] = sum;
    }

    return output;
}
#endif
