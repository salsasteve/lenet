#ifndef WEIGHTS_AND_BIAS_H
#define WEIGHTS_AND_BIAS_H

#include <vector>
#include <cstdint>
struct QuantizedDense{
    int32_t dim1_size;
    int32_t dim2_size;
    std::vector<std::vector<uint16_t>> weights;
    std::vector<uint16_t> biases;
    int weights_zero_points;
    float weights_scale;
    int biases_zero_points;
    float biases_scale;
};

struct Dense {
    int32_t dim1_size;
    int32_t dim2_size;
    std::vector<std::vector<float>> weights;
    std::vector<float> biases;
};

struct Conv2D {
    int Height;
    int Width;
    int Depth;
    int Count;
    int horizontalStride;
    int verticalStride;
    int paddingAmount;
    std::vector<std::vector<std::vector<std::vector<float>>>> weights;
    std::vector<float> biases;
};

extern const QuantizedDense dense_quantized_parameters_1;
extern const Dense dense_parameters_1;
extern const Dense dense_parameters_2;
extern const Dense dense_parameters_3;
extern const Conv2D conv2d_parameters_1;
extern const Conv2D conv2d_parameters_2;
#endif  // WEIGHTS_AND_BIAS_H
