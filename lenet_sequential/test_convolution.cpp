#include <iostream>
#include <vector>
#include "MNISTLoader.h"
#include "config.hpp"
#include <fstream>
// #include <armadillo>

using namespace std;
// using namespace arma;
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
    std::cout << "Kernel height: " << kernelHeight << endl;
    int kernelWidth = kernels[0][0].size();
    std::cout << "Kernel width: " << kernelWidth << endl;
    FeatureMap outputMap(outputHeight, vector<float>(outputWidth, 0));

    std::cout << "Output map dimensions: " << outputMap.size() << "x" << outputMap[0].size() << endl;

    std::cout << "Padded input maps dimensions: " << paddedInputMaps.size() << "x" << paddedInputMaps[0].size() << "x" << paddedInputMaps[0][0].size() << endl;

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
    std::cout << "convolve2d" << endl;
    std::cout << "Output map dimensions: " << outputMap.size() << "x" << outputMap[0].size() << endl;
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

FeatureMaps convolve2dDeep(const FeatureMaps &inputMaps, const DeepKernels &kernels, const vector<float> &biases, int stride, int padding)
{
    if (inputMaps.empty() || kernels.empty() || biases.size() != kernels.size() || inputMaps.size() != kernels[0].size() || stride <= 0 || padding < 0)
    {
        throw std::invalid_argument("Invalid arguments.");
    }

    auto paddedInputMaps = applyPadding(inputMaps, padding);
    int numOutputMaps = kernels.size();
    int outputHeight = (inputMaps[0].size() - kernels[0][0].size() + 2 * padding) / stride + 1;
    int outputWidth = (inputMaps[0][0].size() - kernels[0][0][0].size() + 2 * padding) / stride + 1;
    FeatureMaps outputMaps(numOutputMaps, FeatureMap(outputHeight, vector<float>(outputWidth, 0.0)));

    for (int numOfKernels = 0; numOfKernels < kernels.size(); ++numOfKernels)
    {
        outputMaps[numOfKernels] = convolve2d(paddedInputMaps, kernels[numOfKernels], biases[numOfKernels], stride, outputHeight, outputWidth);
    }

    return outputMaps;
}

// Define the number of kernels, and dimensions
const int num_kernels = 6;
const int height = 5;
const int width = 5;

// Define six 5x5 kernels
float test_kernels[6][5][5] = {
    {// Kernel 1
     {1, 0, 0, 0, 0},
     {0, 1, 0, 0, 0},
     {0, 0, 1, 0, 0},
     {0, 0, 0, 1, 0},
     {0, 0, 0, 0, 1}},
    {// Kernel 2
     {0, 0, 0, 0, 1},
     {0, 0, 0, 1, 0},
     {0, 0, 1, 0, 0},
     {0, 1, 0, 0, 0},
     {1, 0, 0, 0, 0}},
    {// Kernel 3
     {1, 1, 1, 1, 1},
     {1, 1, 1, 1, 1},
     {1, 1, 1, 1, 1},
     {1, 1, 1, 1, 1},
     {1, 1, 1, 1, 1}},
    {// Kernel 4
     {-1, -1, -1, -1, -1},
     {-1, 2, 2, 2, -1},
     {-1, 2, 8, 2, -1},
     {-1, 2, 2, 2, -1},
     {-1, -1, -1, -1, -1}},
    {// Kernel 5
     {0, -1, 0, -1, 0},
     {-1, 0, -1, 0, -1},
     {0, -1, 4, -1, 0},
     {-1, 0, -1, 0, -1},
     {0, -1, 0, -1, 0}},
    {// Kernel 6
     {-2, -1, 0, -1, -2},
     {-1, 0, 1, 0, -1},
     {0, 1, 2, 1, 0},
     {-1, 0, 1, 0, -1},
     {-2, -1, 0, -1, -2}}};

// create test bias vector
float test_biases[6] = {1, 1, 1, 1, 1, 1};

int main()
{

    // LeNet-5 Input Layer configuration
    inputLayerConfig inputLayer = {28, 28, 1, 28 * 28};

    // LeNet-5 Layer 1 configuration
    ConvLayerConfig layer1Config = {5, 5, 1, 6, 1, 1, 2};

    // Create 6 kernels for the first layer of LeNet-5 4D vector with aliases
    Kernels layer1Kernels(layer1Config.kernelDepth, Kernel(layer1Config.kernelHeight, vector<float>(layer1Config.kernelWidth, 0.0)));
    DeepKernels layer1DeepKernels(layer1Config.kernelsCount, layer1Kernels);

    // Initialize the kernels with random values
    // THIS WILL EVENTUALLY BE REPLACED WITH THE TRAINED KERNELS
    for (int k = 0; k < layer1Config.kernelsCount; ++k)
    {
        for (int d = 0; d < layer1Config.kernelDepth; ++d)
        {
            for (int i = 0; i < layer1Config.kernelHeight; ++i)
            {
                for (int j = 0; j < layer1Config.kernelWidth; ++j)
                {
                    layer1DeepKernels[k][d][i][j] = test_kernels[k][i][j];
                }
            }
        }
    }

    // Test the convolve2d
    // FeatureMap convolve2d(
    // const FeatureMaps& paddedInputMaps,  // [channels][height][width]
    // const Kernels& kernels,           // [depth][height][width]
    // float bias,
    // int stride,
    // int outputHeight,
    // int outputWidth)

    // Create a 5x5 input map
    FeatureMap inputMap(5, vector<float>(5, 0.0));
    for (int i = 0; i < 5; ++i)
    {
        for (int j = 0; j < 5; ++j)
        {
            inputMap[i][j] = 1;
        }
    }
    std::cout << "Input map: " << endl;
    for (int i = 0; i < 5; ++i)
    {
        for (int j = 0; j < 5; ++j)
        {
            std::cout << inputMap[i][j] << " ";
        }
        std::cout << endl;
    }

    // Apply padding
    FeatureMaps inputMaps(1, inputMap);
    FeatureMaps paddedInputMaps = applyPadding(inputMaps, 2);
    std::cout << "Padded input maps: " << endl;
    for (int i = 0; i < paddedInputMaps[0].size(); ++i)
    {
        for (int j = 0; j < paddedInputMaps[0][0].size(); ++j)
        {
            std::cout << paddedInputMaps[0][i][j] << " ";
        }
        std::cout << endl;
    }

    // Perform convolution
    FeatureMap outputMap = convolve2d(paddedInputMaps, layer1DeepKernels[2], test_biases[0], 1, 5, 5);

    // Print the output map
    std::cout << "Output map: " << endl;
    for (int i = 0; i < 5; ++i)
    {
        for (int j = 0; j < 5; ++j)
        {
            std::cout << outputMap[i][j] << " ";
        }
        std::cout << endl;
    }

    // Test the convolve2dDeep
    FeatureMaps outputMaps = convolve2dDeep(inputMaps, layer1DeepKernels, vector<float>(test_biases, test_biases + 6), 1, 2);

    // Print the output maps
    std::cout << "Output maps: " << endl;
    for (int m = 0; m < 6; ++m)
    {
        std::cout << "Output map " << m << ":" << endl;
        for (int i = 0; i < 5; ++i)
        {
            for (int j = 0; j < 5; ++j)
            {
                std::cout << outputMaps[m][i][j] << " ";
            }
            std::cout << endl;
        }
    }

    return 0;
}