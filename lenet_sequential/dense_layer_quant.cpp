#ifndef DENSE_LAYER_QUANT_CPP
#define DENSE_LAYER_QUANT_CPP

#include "dense_layer_quant.h"
#include <cstdint>
#include <vector>

std::vector<uint16_t> dense_quant(
     std::vector<float> &input,           
     std::vector<uint16_t> &bias,                 
     std::vector<std::vector<uint16_t>> &weights, 
    int numOutputs)
{
    std::vector<uint16_t> output;
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
