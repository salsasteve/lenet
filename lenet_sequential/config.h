#ifndef CONFIG_H
#define CONFIG_H

#include <string>

extern const std::string conv2d_1_weights;
extern const std::string conv2d_1_bias;
extern const std::string conv2d_2_weights;
extern const std::string conv2d_2_bias;
extern const std::string dense_1_weights;
extern const std::string dense_1_bias;
extern const std::string dense_2_weights;
extern const std::string dense_2_bias;
extern const std::string dense_3_weights;
extern const std::string dense_3_bias;

struct ConvLayerConfig {
    int kernelWidth;
    int kernelHeight;
    int kernelDepth;
    int kernelsCount;
    int horizontalStride;
    int verticalStride;
    int paddingAmount;
};

struct PoolLayerConfig {
    int windowSize;
    int strideLength;
};

struct DenseLayerConfig {
    int inputDimensions;
    int outputDimensions;
};

struct InputLayerConfig {
    int inputWidth;
    int inputHeight;
    int numOfImageForInference;
    int image_size;
};

#endif // CONFIG_H
