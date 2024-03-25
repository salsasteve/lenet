#include "layers/convolution_layer.h"
#include "MNISTLoader.h"
#include <iostream>
#include <vector>




int main() {
    // Parameters for the convolution layer
    const std::string imagesFilename = "../mnist_data/mnist_x_test.bin"; // Path to your MNIST images binary file
    const std::string labelsFilename = "../mnist_data/mnist_y_test.bin"; // Path to your MNIST labels binary file
    size_t inputHeight = 28;
    size_t inputWidth = 28;
    size_t inputDepth = 10;  // Number of images for inference
    size_t total_images = 10000;
    size_t filterHeight= 5;
    size_t filterWidth = 5;
    size_t horizontalStride = 1;
    size_t verticalStride = 1;
    size_t layer1NumFilters = 6;
    size_t paddingHeight = 2; // Padding around the input
    size_t paddingWidth = 2; // Padding around the input

    // Updated input dimensions including padding
    size_t paddedInputHeight = inputHeight + 2 * paddingHeight;
    size_t paddedInputWidth = inputWidth + 2 * paddingWidth;

    // check the dimensions
    std::cout << "Input Dimensions: " << inputHeight << "x" << inputWidth << "x" << inputDepth << std::endl;
    std::cout << "Filter Dimensions: " << filterHeight << "x" << filterWidth << std::endl;
    std::cout << "Stride: " << horizontalStride << "x" << verticalStride << std::endl;
    std::cout << "Padding: " << paddingHeight << "x" << paddingWidth << std::endl;
    std::cout << "Number of Filters: " << layer1NumFilters << std::endl;


    // Load the MNIST images and labels from binary files
    std::vector<float> images; // Vector to hold image data
    // loadMNISTBinary(imagesFilename, images, 1, inputHeight * inputWidth); // Load only one image for simplicity
    loadMNISTBinary(imagesFilename, images, total_images, inputHeight * inputWidth);
    std::vector<unsigned char> labels = readMNISTLabels(labelsFilename, 1); // Load the corresponding label

    // images dimensions
    std::cout << "Images Dimensions Before Matrix: " << images.size() << std::endl;
    
    // create 3d vector to hold the images
    std::vector<std::vector<std::vector<float>>> input(inputDepth, 
        std::vector<std::vector<float>>(inputHeight, std::vector<float>(inputWidth, 0.0)));

    input = convertTo3DVector(images, inputDepth, inputHeight * inputWidth);
    

    // print dimensions
    std::cout << "Input Dimensions: " << input.size() << "x" << input[0].size() << "x" << input[0][0].size() << std::endl;


    // // Create an instance of ConvolutionLayer
    // ConvolutionLayer convLayer(inputHeight, inputWidth, inputDepth,
    //                            filterHeight, filterWidth,
    //                            horizontalStride, verticalStride,
    //                            paddingHeight, paddingWidth,
    //                            layer1NumFilters);

    // // Prepare an output structure (initializing with zeros)
    // size_t convLayer1OutputHeight = (paddedInputHeight - filterHeight) / verticalStride + 1;
    // size_t convLayer1OutputWidth = (paddedInputWidth - filterWidth) / horizontalStride + 1;

    // std::cout << "convLayer1 padded Input Dimensions: " << inputDepth << "x" << paddedInputHeight << "x" << paddedInputWidth << std::endl;
    // std::cout << "convLayer1 Output Dimensions: " << layer1NumFilters << "x" << convLayer1OutputHeight << "x" << convLayer1OutputWidth << std::endl;

    // std::vector<std::vector<std::vector<float>>> output(layer1NumFilters, 
    //     std::vector<std::vector<float>>(convLayer1OutputHeight, std::vector<float>(convLayer1OutputWidth, 0.0)));

    // // Run the forward pass
    // convLayer.Forward(input, output);
    
    // // Print the output
    // std::cout << "Output of the Convolution Layer:" << std::endl;
    
    // // print the output dimensions
    // std::cout << "Output Dimensions: " << output.size() << "x" << output[0].size() << "x" << output[0][0].size() << std::endl;
    // // print filters dimensions
    // std::vector<std::vector<std::vector<std::vector<float>>> > filters = convLayer.getFilters();
    // std::cout << "Filters Dimensions: " << filters.size() << "x" << filters[0].size() << "x" << filters[0][0].size() << "x" << filters[0][0][0].size() << std::endl;

    return 0;
}
