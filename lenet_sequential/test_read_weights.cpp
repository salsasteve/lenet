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




int main() {
    // Input images
    const std::string imagesFilename = "../mnist_data/mnist_x_test.bin"; // Path to your MNIST images binary file
    const std::string labelsFilename = "../mnist_data/mnist_y_test.bin"; // Path to your MNIST labels binary file
    
    int inputHeight = 28;
    int inputWidth = 28;
    int total_images = 10000;// Number of images available in the dataset
    int numOfImageForInference = 1;  // Number of used images for inference
        // LeNet-5 Input Layer configuration
    inputLayerConfig inputLayer = {28, 28, 1, 28 * 28};

    // LeNet-5 Layer 1 configuration
    ConvLayerConfig layer1Config = {5, 5, 1, 6, 1, 1, 2};


    // Load the MNIST images and labels from binary files
    std::vector<float> allImages; // Vector to hold image data
    loadMNISTBinary(imagesFilename, allImages, total_images, inputLayer.image_size);

    // create 3d vector to hold the images with aliases
    vector<Image> input(inputLayer.numOfImageForInference, Image(inputLayer.inputHeight, vector<float>(inputLayer.inputWidth, 0.0)));

    // check dimensions
    std::cout << "Input dimensions: " << input.size() << "x" << input[0].size() << "x" << input[0][0].size() << endl;

    input = convertTo3DVector(allImages, inputLayer.numOfImageForInference, inputLayer.image_size);

    // Create 6 kernels for the first layer of LeNet-5 4D vector with aliases
    Kernels layer1Kernels(layer1Config.kernelDepth, 
                          Kernel(layer1Config.kernelHeight, 
                                 vector<float>(layer1Config.kernelWidth, 0.0)));
    DeepKernels layer1DeepKernels(layer1Config.kernelsCount, layer1Kernels);
    
    // Check the dimensions of the kernels
    std::cout << "Kernel dimensions: " << layer1DeepKernels.size() << "x" << layer1DeepKernels[0].size() << "x" << layer1DeepKernels[0][0].size() << "x" << layer1DeepKernels[0][0][0].size() << endl;

    string filename = "../read_model/parameters/conv2d_1_weights.bin";
    layer1DeepKernels = LoadConv2DWeights(filename,
                                          layer1Config.kernelsCount,
                                          layer1Config.kernelDepth,
                                          layer1Config.kernelHeight,
                                          layer1Config.kernelWidth);
 

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