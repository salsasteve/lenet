#include <iostream>
#include <vector>
#include "../MNISTLoader.h"
#include "config.hpp"


using namespace std;
using Image = vector<vector<float>>;
using Batch = vector<Image>;
using Kernel = vector<vector<float>>;
using Kernels = vector<Kernel>;
using DeepKernels = vector<Kernels>;
using FeatureMap = vector<vector<float>>;
using FeatureMaps = vector<FeatureMap>;



/**
 * Performs 2D convolution on an input image using multiple kernels (filters),
 * then adds a bias to each output feature map. This operation is fundamental
 * in the convolutional layer of a neural network, such as in LeNet-5 architecture,
 * especially during the forward pass.
 *
 * Parameters:
 *  - image: A 2D vector of floats representing the grayscale input image. 
 *           The dimensions are assumed to be [imageHeight][imageWidth].
 *  - kernels: A 3D vector where each 2D vector (element) represents a 
 *             convolutional kernel or filter. The expected dimensions are 
 *             [numberOfKernels][kernelHeight][kernelWidth].
 *  - biases: A vector of floats where each element is a bias value associated 
 *            with one convolutional kernel.
 *  - stride: The number of pixels by which the kernel moves over the image. 
 *            This parameter controls the spatial resolution of the output feature maps.
 *  - padding: The number of pixels added to the periphery of the input image. 
 *             This is used to control the size of the output feature maps.
 *
 * Returns:
 *  - A 3D vector of floats representing the output feature maps of the convolution. 
 *    Each 2D vector (element) is one feature map corresponding to one kernel, 
 *    with dimensions [numberOfKernels][outputHeight][outputWidth]. 
 *    These feature maps highlight different features of the input image, 
 *    depending on the patterns of the kernels.
 *
 * Usage in LeNet-5 Forward Pass:
 *  - This function can be used in the forward pass of the LeNet-5 network 
 *    to perform convolutions on the input layer or on the output of previous layers.
 *  - In LeNet-5, the first layer often employs convolution to extract basic image features 
 *    like edges and gradients. This function would be applied to the raw input image.
 *  - Subsequent convolutional layers would use this function to further refine and 
 *    extract more complex patterns from the initial feature maps, contributing to 
 *    higher-level feature detection essential for successful image classification.
 *  - The output of this function (after applying activation functions, pooling, etc., 
 *    in subsequent steps) propagates through the network, forming the basis for the 
 *    final classification decision in LeNet-5.
 */

vector<vector<vector<float>>> convolve2d(const vector<vector<float>>& image, 
                                         const vector<vector<vector<float>>>& kernels, 
                                         const vector<float>& biases, 
                                         const int stride, 
                                         const int padding) {
    int imageHeight = image.size();
    int imageWidth = image[0].size();
    int numberOfKernels = kernels.size();
    int kernelHeight = kernels[0].size();
    int kernelWidth = kernels[0][0].size();
    
    // Output dimensions
    int outputHeight = (imageHeight - kernelHeight + 2 * padding) / stride + 1;
    int outputWidth = (imageWidth - kernelWidth + 2 * padding) / stride + 1;

    // Initialize output matrix with zeros
    vector<vector<vector<float>>> output(numberOfKernels, vector<vector<float>>(outputHeight, vector<float>(outputWidth, 0)));

    // Apply padding to the image
    vector<vector<float>> paddedImage(imageHeight + 2 * padding, 
                                      vector<float>(imageWidth + 2 * padding, 0));
    for (int i = 0; i < imageHeight; ++i) {
        for (int j = 0; j < imageWidth; ++j) {
            paddedImage[i + padding][j + padding] = image[i][j];
        }
    }

    // Perform the convolution operation for each kernel
    for (int k = 0; k < numberOfKernels; ++k) {
        for (int y = 0; y < outputHeight; ++y) {
            for (int x = 0; x < outputWidth; ++x) {
                float sum = 0.0;
                for (int i = 0; i < kernelHeight; ++i) {
                    for (int j = 0; j < kernelWidth; ++j) {
                        sum += kernels[k][i][j] * paddedImage[y * stride + i][x * stride + j];
                    }
                }
                output[k][y][x] = sum + biases[k];  // Add the bias for the current kernel
            }
        }
    }

    return output;
}

vector<vector<vector<float>>> applyPadding(const vector<vector<vector<float>>>& inputMaps, int padding) {
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
    vector<vector<vector<float>>> paddedMaps(numMaps, vector<vector<float>>(height + 2 * padding, vector<float>(width + 2 * padding, 0)));

    for (int m = 0; m < numMaps; ++m) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                paddedMaps[m][i + padding][j + padding] = inputMaps[m][i][j];
            }
        }
    }

    return paddedMaps;
}

vector<vector<float>> singleMapConvolution(
    const vector<vector<vector<float>>>& paddedInputMaps, 
    const vector<vector<vector<float>>>& kernel, 
    float bias, 
    int stride, 
    int outputHeight, 
    int outputWidth) {
    
    if (paddedInputMaps.empty() || kernel.empty() || kernel[0].empty()) {
        throw std::invalid_argument("Padded input maps and kernel cannot be empty.");
    }
    if (stride <= 0) {
        throw std::invalid_argument("Stride must be positive.");
    }
    if (outputHeight <= 0 || outputWidth <= 0) {
        throw std::invalid_argument("Output dimensions must be positive.");
    }
    
    int kernelHeight = kernel[0].size();
    int kernelWidth = kernel[0][0].size();
    vector<vector<float>> outputMap(outputHeight, vector<float>(outputWidth, 0));

    for (int y = 0; y < outputHeight; ++y) {
        for (int x = 0; x < outputWidth; ++x) {
            float sum = 0.0;
            for (int m = 0; m < paddedInputMaps.size(); ++m) {
                for (int i = 0; i < kernelHeight; ++i) {
                    for (int j = 0; j < kernelWidth; ++j) {
                        sum += kernel[m][i][j] * paddedInputMaps[m][y * stride + i][x * stride + j];
                    }
                }
            }
            outputMap[y][x] = sum + bias; // Add the bias
        }
    }

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
        throw std::invalid_argument("Number of biases must match number of output feature maps.");
    }
    if (inputMaps[0].size() != kernels[0][0].size()) {
        std::cout << "Input maps depth: " << inputMaps[0].size() << endl;
        std::cout << "Kernel depth: " << kernels[0][0].size() << endl;
        throw std::invalid_argument("Mismatch between input maps depth and kernel depth.");
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
    int outputHeight = (inputMaps[0].size() - kernels[0][0].size() + 2 * padding) / stride + 1;
    int outputWidth = (inputMaps[0][0].size() - kernels[0][0][0].size() + 2 * padding) / stride + 1;

    // Initialize output maps
    vector<vector<vector<float>>> outputMaps(numOutputMaps);

    // Perform convolution on each output feature map
    for (int n = 0; n < numOutputMaps; ++n) {
        outputMaps[n] = singleMapConvolution(paddedInputMaps, kernels[n], biases[n], stride, outputHeight, outputWidth);
    }

    return outputMaps;
}



int main() {
    // Input images
    const std::string imagesFilename = "../../mnist_data/mnist_x_test.bin"; // Path to your MNIST images binary file
    const std::string labelsFilename = "../../mnist_data/mnist_y_test.bin"; // Path to your MNIST labels binary file
    
    int inputHeight = 28;
    int inputWidth = 28;
    int total_images = 10000;// Number of images available in the dataset
    int numOfImageForInference = 10;  // Number of used images for inference
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
    Kernels layer1Kernels(layer1Config.kernelsCount, Kernel(layer1Config.kernelHeight, vector<float>(layer1Config.kernelWidth, 0.0)));
    DeepKernels layer1DeepKernels(layer1Config.kernelDepth, layer1Kernels);

    // Initialize the kernels with random values
    // THIS WILL EVENTUALLY BE REPLACED WITH THE TRAINED KERNELS
    for (int k = 0; k < layer1Config.kernelsCount; ++k) {
        for (int i = 0; i < layer1Config.kernelHeight; ++i) {
            for (int j = 0; j < layer1Config.kernelWidth; ++j) {
                layer1DeepKernels[k][0][i][j] = (rand() % 100) / 100.0;  // Random float between 0 and 1
            }
        }
    }

    // Create biases for each kernel
    vector<float> biases(layer1Config.kernelsCount, 0.0);
    // Initialize the biases with random values
    // THIS WILL EVENTUALLY BE REPLACED WITH THE TRAINED BIASES
    for (int k = 0; k < layer1Config.kernelsCount; ++k) {
        biases[k] = (rand() % 100) / 100.0;  // Random float between 0 and 1
    }
    
    // Perform 2D deep convolution on the input image
    FeatureMaps layer1FeatureMaps = convolve2dDeep(input, layer1DeepKernels, biases, layer1Config.horizontalStride, layer1Config.paddingAmount);

    // Print the output feature maps dimensions
    cout << "Output feature maps:" << endl;
    cout << "Number of output feature maps: " << layer1FeatureMaps.size() << endl;
    cout << "Output feature map dimensions: " << layer1FeatureMaps[0].size() << "x" << layer1FeatureMaps[0][0].size() << endl;

    return 0;
}
