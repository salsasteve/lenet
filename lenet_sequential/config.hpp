#ifndef CONFIG_HPP
#define CONFIG_HPP

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

struct inputLayerConfig {
    int inputWidth;
    int inputHeight;
    int numOfImageForInference;
    int image_size;
};

#endif // CONFIG_HPP
