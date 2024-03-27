#include <iostream>
#include <vector>
#include "MNISTLoader.h"
#include "config.hpp"
#include "read_model_weights.h"
#include "activations.h"
#include "pooling.h"
#include <fstream>

using namespace std;
using Image = vector<vector<float>>;
using Batch = vector<Image>;
// 2D vector to hold the kernels for each layer
// [height][width]
using Kernel = vector<vector<float>>;
// 3D vector to hold the kernels for each layer
// [number of kernels][height][width
using Kernels = vector<Kernel>;
// 4D vector to hold the kernels for each layer
// [number of kernels][depth][height][width]
// depth is the number of input feature maps
// number of kernels is the number of output feature maps
using DeepKernels = vector<Kernels>;
// 2D vector to hold the feature maps for each layer
// [height][width]
using FeatureMap = vector<vector<float>>;
// 3D vector to hold the feature maps for each layer
// [channels][height][width]
// channels is the number of output feature maps
using FeatureMaps = vector<FeatureMap>;

FeatureMaps applyPadding(const FeatureMaps &inputMaps, int padding)
{
    if (inputMaps.empty() || inputMaps[0].empty() || inputMaps[0][0].empty())
    {
        // Handle error or empty case
        throw std::invalid_argument("Input maps are empty or improperly structured.");
    }
    if (padding < 0)
    {
        throw std::invalid_argument("Padding cannot be negative.");
    }
    int numMaps = inputMaps.size();
    for (const auto &map : inputMaps)
    {
        if (map.size() != inputMaps[0].size() || map[0].size() != inputMaps[0][0].size())
        {
            throw std::invalid_argument("All input maps must have the same dimensions.");
        }
    }

    int height = inputMaps[0].size();
    int width = inputMaps[0][0].size();
    FeatureMaps paddedMaps(numMaps, vector<vector<float>>(height + 2 * padding, vector<float>(width + 2 * padding, 0)));

    for (int m = 0; m < numMaps; ++m)
    {
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                paddedMaps[m][i + padding][j + padding] = inputMaps[m][i][j];
            }
        }
    }
    std::cout << "Padded maps dimensions: " << paddedMaps.size() << "x" << paddedMaps[0].size() << "x" << paddedMaps[0][0].size() << endl;
    return paddedMaps;
}

FeatureMap convolve2d(
    const FeatureMaps &paddedInputMaps, // [channels][height][width]
    const Kernels &kernels,             // [depth][height][width]
    float bias,
    int stride,
    int outputHeight,
    int outputWidth)
{

    if (paddedInputMaps.empty() || kernels.empty() || kernels[0].empty())
    {
        throw std::invalid_argument("Padded input maps and kernel cannot be empty.");
    }
    if (stride <= 0)
    {
        throw std::invalid_argument("Stride must be positive.");
    }
    if (outputHeight <= 0 || outputWidth <= 0)
    {
        throw std::invalid_argument("Output dimensions must be positive.");
    }
    if (paddedInputMaps.size() != kernels.size())
    {
        throw std::invalid_argument("Mismatch between input maps channels and kernel depth.");
    }

    int kernelHeight = kernels[0].size();
    int kernelWidth = kernels[0][0].size();
    FeatureMap outputMap(outputHeight, vector<float>(outputWidth, 0));

    for (int y = 0; y < outputHeight; ++y)
    {
        for (int x = 0; x < outputWidth; ++x)
        {
            float sum = 0.0;
            for (int m = 0; m < paddedInputMaps.size(); ++m)
            {
                for (int i = 0; i < kernelHeight; ++i)
                {
                    for (int j = 0; j < kernelWidth; ++j)
                    {
                        sum += kernels[m][i][j] * paddedInputMaps[m][y * stride + i][x * stride + j];
                    }
                }
            }
            outputMap[y][x] = sum + bias; // Add the bias
        }
    }
    return outputMap;
}

void validateImageDimensions(const vector<vector<vector<float>>> &images, int expectedHeight, int expectedWidth)
{
    for (const auto &image : images)
    {
        if (image.size() != expectedHeight)
        {
            throw std::runtime_error("Unexpected image height.");
        }
        for (const auto &row : image)
        {
            if (row.size() != expectedWidth)
            {
                throw std::runtime_error("Unexpected image width.");
            }
        }
    }
}

FeatureMaps convolve2dDeep(
    const FeatureMaps &inputMaps,
    const DeepKernels &kernels,
    const vector<float> &biases,
    const int stride,
    const int padding)
{

    // Checks on input parameters
    if (inputMaps.empty() || kernels.empty())
    {
        throw std::invalid_argument("Input maps and kernels cannot be empty.");
    }
    if (biases.size() != kernels.size())
    {
        std::cout << "Number of biases: " << biases.size() << endl;
        std::cout << "Number of output feature maps: " << kernels.size() << endl;
        throw std::invalid_argument("Number of biases must match number of output feature maps.");
    }
    if (inputMaps.size() != kernels[0].size())
    {
        std::cout << "Input maps channels " << inputMaps.size() << endl;
        std::cout << "Kernel depth: " << kernels.size() << endl;
        throw std::invalid_argument("Mismatch between input maps channels and kernel depth.");
    }
    if (stride <= 0)
    {
        throw std::invalid_argument("Stride must be positive.");
    }
    if (padding < 0)
    {
        throw std::invalid_argument("Padding cannot be negative.");
    }

    // Initialize and apply padding
    auto paddedInputMaps = applyPadding(inputMaps, padding);

    // Calculate output dimensions
    int numOutputMaps = kernels.size();
    std::cout << "Number of output maps: " << numOutputMaps << endl;
    int outputHeight = (inputMaps[0].size() - kernels[0][0].size() + 2 * padding) / stride + 1;
    std::cout << "Output height: " << outputHeight << endl;
    int outputWidth = (inputMaps[0][0].size() - kernels[0][0][0].size() + 2 * padding) / stride + 1;
    std::cout << "Output width: " << outputWidth << endl;

    // Initialize output maps
    // vector<vector<vector<float>>> outputMaps(numOutputMaps);
    FeatureMaps outputMaps(numOutputMaps, FeatureMap(outputHeight, vector<float>(outputWidth, 0.0)));

    // Perform convolution on each output feature map
    std::cout << "Biases count: " << biases.size() << endl;
    std::cout << "Kernels count: " << kernels.size() << endl;

    for (int numOfKernels = 0; numOfKernels < kernels.size(); ++numOfKernels)
    {
        outputMaps[numOfKernels] = convolve2d(paddedInputMaps, kernels[numOfKernels], biases[numOfKernels], stride, outputHeight, outputWidth);
    }
    std::cout << "convolve2dDeep" << endl;
    std::cout << "Output maps dimensions: " << outputMaps.size() << "x" << outputMaps[0].size() << "x" << outputMaps[0][0].size() << endl;
    return outputMaps;
}

vector<float> flatten(const FeatureMaps &featureMaps)
{
    vector<float> flattened;
    for (const auto &map : featureMaps)
    {
        for (const auto &row : map)
        {
            for (auto val : row)
            {
                flattened.push_back(val);
            }
        }
    }
    return flattened;
}

int main()
{
    // LeNet-5 Input Layer configuration
    inputLayerConfig inputLayer = {28, 28, 1, 28 * 28};
    // LeNet-5 Layer 1 configuration
    ConvLayerConfig layer1Config = {5, 5, 1, 6, 1, 1, 2};

    // create 3d vector to hold the images with aliases
    vector<Image> input(inputLayer.numOfImageForInference, Image(inputLayer.inputHeight, vector<float>(inputLayer.inputWidth, 0.0)));

    input = getMNISTImages();
    validateImageDimensions(input, inputLayer.inputHeight, inputLayer.inputWidth);

    // Create 6 kernels for the first layer of LeNet-5 4D vector with aliases
    Kernel layer1Kernel(layer1Config.kernelHeight, vector<float>(layer1Config.kernelWidth, 0.0));
    Kernels layer1Kernels(layer1Config.kernelDepth, layer1Kernel);
    DeepKernels layer1DeepKernels(layer1Config.kernelsCount, layer1Kernels);

    // Check the dimensions of the kernels
    std::cout << "Kernel dimensions: " << layer1DeepKernels.size() << "x" << layer1DeepKernels[0].size() << "x" << layer1DeepKernels[0][0].size() << "x" << layer1DeepKernels[0][0][0].size() << endl;

    string conv2d_1_weights = "../read_model/parameters/conv2d_1_weights.bin";
    layer1DeepKernels = LoadConv2DWeights(conv2d_1_weights,
                                          layer1Config.kernelsCount,
                                          layer1Config.kernelDepth,
                                          layer1Config.kernelHeight,
                                          layer1Config.kernelWidth);

    string conv2d_1_bias = "../read_model/parameters/conv2d_1_bias.bin";
    vector<float> biases(layer1Config.kernelsCount);
    biases = LoadBias(conv2d_1_bias, layer1Config.kernelsCount);

    std::cout << "Kernel dimensions: " << layer1DeepKernels.size() << "x" << layer1DeepKernels[0].size() << "x" << layer1DeepKernels[0][0].size() << "x" << layer1DeepKernels[0][0][0].size() << endl;

    FeatureMaps layer1FeatureMaps = convolve2dDeep(input, layer1DeepKernels, biases, layer1Config.horizontalStride, layer1Config.paddingAmount);
    FeatureMaps layer1activatedFeatureMaps = tanh3D(layer1FeatureMaps);
    FeatureMaps layer1PooledFeatureMaps = averagePooling3D(layer1activatedFeatureMaps, 2, 2);
    
    ConvLayerConfig layer2Config = {5, 5, 6, 16, 1, 1, 0};

    // Create 16 kernels for the second layer of LeNet-5 4D vector with aliases
    Kernel layer2Kernel(layer2Config.kernelHeight, vector<float>(layer2Config.kernelWidth, 0.0));
    Kernels layer2Kernels(layer2Config.kernelDepth, layer2Kernel);
    DeepKernels layer2DeepKernels(layer2Config.kernelsCount, layer2Kernels);

    string conv2d_2_weights = "../read_model/parameters/conv2d_2_weights.bin";
    layer2DeepKernels = LoadConv2DWeights(conv2d_2_weights,
                                          layer2Config.kernelsCount,
                                          layer2Config.kernelDepth,
                                          layer2Config.kernelHeight,
                                          layer2Config.kernelWidth);
    string conv2d_2_bias = "../read_model/parameters/conv2d_2_bias.bin";
    biases = LoadBias(conv2d_2_bias, layer2Config.kernelsCount);

    FeatureMaps layer2FeatureMaps = convolve2dDeep(layer1PooledFeatureMaps, layer2DeepKernels, biases, layer2Config.horizontalStride, layer2Config.paddingAmount);
    // Check the dimensions of the feature maps
    std::cout << "Layer 2 feature maps dimensions: " << layer2FeatureMaps.size() << "x" << layer2FeatureMaps[0].size() << "x" << layer2FeatureMaps[0][0].size() << endl;
    FeatureMaps layer2activatedFeatureMaps = tanh3D(layer2FeatureMaps);
    FeatureMaps layer2PooledFeatureMaps = averagePooling3D(layer2activatedFeatureMaps, 2, 2);

    // Check the dimensions of the feature maps
    std::cout << "Layer 2 feature maps dimensions: " << layer2PooledFeatureMaps.size() << "x" << layer2PooledFeatureMaps[0].size() << "x" << layer2PooledFeatureMaps[0][0].size() << endl;

    // Flatten the feature maps
    vector<float> flattenedFeatures = flatten(layer2PooledFeatureMaps);
    std::cout << "Flattened features count: " << flattenedFeatures.size() << endl;

    // Load the weights for the first dense layer


    return 0;
}