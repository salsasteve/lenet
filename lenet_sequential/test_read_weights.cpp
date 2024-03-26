#include <iostream>
#include <vector>
#include "MNISTLoader.h"
#include "config.hpp"
#include "read_model_weights.h"
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


FeatureMaps applyPadding(const FeatureMaps& inputMaps, int padding) {
    if (inputMaps.empty() || inputMaps[0].empty() || inputMaps[0][0].empty()) {
        // Handle error or empty case
        throw std::invalid_argument("Input maps are empty or improperly structured.");
    }
    if (padding < 0) {
        throw std::invalid_argument("Padding cannot be negative.");
    }
    int numMaps = inputMaps.size();
    for (const auto& map : inputMaps) {
        if (map.size() != inputMaps[0].size() || map[0].size() != inputMaps[0][0].size()) {
            throw std::invalid_argument("All input maps must have the same dimensions.");
        }
    }

    int height = inputMaps[0].size();
    int width = inputMaps[0][0].size();
    FeatureMaps paddedMaps(numMaps, vector<vector<float>>(height + 2 * padding, vector<float>(width + 2 * padding, 0)));

    for (int m = 0; m < numMaps; ++m) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                paddedMaps[m][i + padding][j + padding] = inputMaps[m][i][j];
            }
        }
    }
    std::cout << "Padded maps dimensions: " << paddedMaps.size() << "x" << paddedMaps[0].size() << "x" << paddedMaps[0][0].size() << endl;
    return paddedMaps;
}

FeatureMap convolve2d(
    const FeatureMaps& paddedInputMaps,  // [channels][height][width]
    const Kernels& kernels,           // [depth][height][width]
    float bias, 
    int stride, 
    int outputHeight, 
    int outputWidth) {
    
    if (paddedInputMaps.empty() || kernels.empty() || kernels[0].empty()) {
        throw std::invalid_argument("Padded input maps and kernel cannot be empty.");
    }
    if (stride <= 0) {
        throw std::invalid_argument("Stride must be positive.");
    }
    if (outputHeight <= 0 || outputWidth <= 0) {
        throw std::invalid_argument("Output dimensions must be positive.");
    }
    if (paddedInputMaps.size() != kernels.size()) {
        throw std::invalid_argument("Mismatch between input maps channels and kernel depth.");
    }
    
    int kernelHeight = kernels[0].size();
    std::cout << "Kernel height: " << kernelHeight << endl;
    int kernelWidth = kernels[0][0].size();
    std::cout << "Kernel width: " << kernelWidth << endl;
    FeatureMap outputMap(outputHeight, vector<float>(outputWidth, 0));

    std::cout << "Output map dimensions: " << outputMap.size() << "x" << outputMap[0].size() << endl;

    std::cout << "Padded input maps dimensions: " << paddedInputMaps.size() << "x" << paddedInputMaps[0].size() << "x" << paddedInputMaps[0][0].size() << endl;

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
    std::cout << "convolve2d" << endl;
    std::cout << "Output map dimensions: " << outputMap.size() << "x" << outputMap[0].size() << endl;
    return outputMap;
}

void validateImageDimensions(const vector<vector<vector<float>>>& images, int expectedHeight, int expectedWidth) {
    for (const auto& image : images) {
        if (image.size() != expectedHeight) {
            throw std::runtime_error("Unexpected image height.");
        }
        for (const auto& row : image) {
            if (row.size() != expectedWidth) {
                throw std::runtime_error("Unexpected image width.");
            }
        }
    }
}

FeatureMaps convolve2dDeep(
    const FeatureMaps& inputMaps, 
    const DeepKernels& kernels, 
    const vector<float>& biases, 
    const int stride, 
    const int padding) {

    // Checks on input parameters
    if (inputMaps.empty() || kernels.empty()) {
        throw std::invalid_argument("Input maps and kernels cannot be empty.");
    }
    if (biases.size() != kernels.size()) {
        std::cout << "Number of biases: " << biases.size() << endl;
        std::cout << "Number of output feature maps: " << kernels.size() << endl;
        throw std::invalid_argument("Number of biases must match number of output feature maps.");
    }
    if (inputMaps.size() != kernels[0].size()) {
        std::cout << "Input maps channels " << inputMaps.size() << endl;
        std::cout << "Kernel depth: " << kernels.size() << endl;
        throw std::invalid_argument("Mismatch between input maps channels and kernel depth.");
    }
    if (stride <= 0) {
        throw std::invalid_argument("Stride must be positive.");
    }
    if (padding < 0) {
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

    for (int numOfKernels = 0; numOfKernels < kernels.size(); ++numOfKernels) {
        std::cout << "numOfKernels: " << numOfKernels << endl;
        outputMaps[numOfKernels] = convolve2d(paddedInputMaps, kernels[numOfKernels], biases[numOfKernels], stride, outputHeight, outputWidth);
    }
    std::cout << "convolve2dDeep" << endl;
    std::cout << "Output maps dimensions: " << outputMaps.size() << "x" << outputMaps[0].size() << "x" << outputMaps[0][0].size() << endl;
    return outputMaps;
}

int main() {
    // Input images
    const std::string imagesFilename = "../mnist_data/mnist_x_test.bin"; // Path to your MNIST images binary file
    const std::string labelsFilename = "../mnist_data/mnist_y_test.bin"; // Path to your MNIST labels binary file
    
    int inputHeight = 28;
    int inputWidth = 28;
    int total_images = 10000;// Number of images available in the dataset
    int numOfImageForInference = 1;  // Number of used images for inference
    // LeNet-5 Input Layer configuration
    inputLayerConfig inputLayer = {inputHeight, inputWidth, numOfImageForInference , inputHeight * inputWidth};
    int layer1KernelHeight= 5;
    int layer1KernelWidth = 5;
    int layer1KernelDepth = 1;
    int layer1KernelCount = 6;
    int layer1HorizontalStride = 1;
    int layer1VerticalStride = 1;
    int layer1PaddingAmount = 2; // Padding around the input
    // LeNet-5 Layer 1 configuration
    ConvLayerConfig layer1Config = {layer1KernelHeight, 
                                    layer1KernelWidth, 
                                    layer1KernelDepth, 
                                    layer1KernelCount, 
                                    layer1HorizontalStride, 
                                    layer1VerticalStride, 
                                    layer1PaddingAmount};


    // Load the MNIST images and labels from binary files
    std::vector<float> allImages; // Vector to hold image data
    loadMNISTBinary(imagesFilename, allImages, total_images, inputLayer.image_size);

    // create 3d vector to hold the images with aliases
    vector<Image> input(inputLayer.numOfImageForInference, Image(inputLayer.inputHeight, vector<float>(inputLayer.inputWidth, 0.0)));

    // check dimensions
    std::cout << "Input dimensions: " << input.size() << "x" << input[0].size() << "x" << input[0][0].size() << endl;

    input = convertTo3DVector(allImages, inputLayer.numOfImageForInference, inputLayer.image_size);
    validateImageDimensions(input, inputLayer.inputHeight, inputLayer.inputWidth);

    // Create 6 kernels for the first layer of LeNet-5 4D vector with aliases
    Kernels layer1Kernels(layer1Config.kernelDepth, 
                          Kernel(layer1Config.kernelHeight, 
                                 vector<float>(layer1Config.kernelWidth, 0.0)));
    DeepKernels layer1DeepKernels(layer1Config.kernelsCount, layer1Kernels);
    
    // Check the dimensions of the kernels
    std::cout << "Kernel dimensions: " << layer1DeepKernels.size() << "x" << layer1DeepKernels[0].size() << "x" << layer1DeepKernels[0][0].size() << "x" << layer1DeepKernels[0][0][0].size() << endl;

    string filename = "../read_model/parameters/conv2d_1_weights.bin";
    layer1DeepKernels = LoadConv2DWeights(filename,
                                          layer1KernelCount,
                                          layer1KernelDepth,
                                          layer1KernelWidth,
                                          layer1KernelHeight);
 

    string filename_3 = "../read_model/parameters/conv2d_1_bias.bin";
    vector<float> biases(layer1Config.kernelsCount);
    biases = LoadBias(filename_3, 6);


    std::cout << "Kernel dimensions: " << layer1DeepKernels.size() << "x" << layer1DeepKernels[0].size() << "x" << layer1DeepKernels[0][0].size() << "x" << layer1DeepKernels[0][0][0].size() << endl;

    // Create biases for each kernel
    if (biases.size() != layer1Config.kernelsCount) {
        throw std::invalid_argument("Number of biases must match number of kernels.");
    }
   
    FeatureMaps layer1FeatureMaps = convolve2dDeep(input, layer1DeepKernels, biases, layer1Config.horizontalStride, layer1Config.paddingAmount);

    // save the feature maps to a file as binary
    std::ofstream file("feature_maps.bin", std::ios::binary);

    if (file.is_open()) {
        for (int i = 0; i < layer1FeatureMaps.size(); ++i) {
            for (int j = 0; j < layer1FeatureMaps[0].size(); ++j) {
                for (int k = 0; k < layer1FeatureMaps[0][0].size(); ++k) {
                    file.write(reinterpret_cast<char*>(&layer1FeatureMaps[i][j][k]), sizeof(float));
                }
            }
        }
        file.close();
    } else {
        std::cerr << "Could not open the file." << std::endl;
    }


    return 0;
}