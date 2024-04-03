#ifndef DENSE_1_WEIGHTS_QUANTIZED_H
#define DENSE_1_WEIGHTS_QUANTIZED_H

#include <cstdint>

struct QuantizedDense {
    int32_t dim1_size;
    int32_t dim2_size;
    const uint16_t* weights;
    const uint16_t* bias;
};

extern const QuantizedDense dense_parameters_1;

#endif  // DENSE_1_WEIGHTS_QUANTIZED_H
