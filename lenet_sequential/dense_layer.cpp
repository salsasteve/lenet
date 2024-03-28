#ifndef DENSE_LAYER_CPP
#define DENSE_LAYER_CPP

#include "dense_layer.h"
#include <vector>

std::vector<float> dense(
    std::vector<float> &input,           
    std::vector<float> &bias,                 
    std::vector<std::vector<float>> &weights, 
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

#endif
