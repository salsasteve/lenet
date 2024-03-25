#include "MNISTLoader.h"
#include "DebugTools.h"
#include "Convolution.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib> // For srand() and rand()
#include <ctime>   // For time()
#include <random>
#include <cmath>
#include <layers/convolution_layer.h>

// Function to create a kernel of given size with values from a chosen distribution
std::vector<std::vector<float>> createDistributedKernel(int rows, int cols, const std::string& distributionType, double param1, double param2) {
    std::vector<std::vector<float>> kernel(rows, std::vector<float>(cols));
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()

    if (distributionType == "uniform") {
        std::uniform_real_distribution<> dis(param1, param2);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                kernel[i][j] = dis(gen);
            }
        }
    } else if (distributionType == "normal") {
        std::normal_distribution<> dis(param1, param2);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                kernel[i][j] = dis(gen);
            }
        }
    }
    
    return kernel;
}

// Function to print the kernel
void printKernel(const std::vector<std::vector<float>>& kernel) {
    for(const auto &row : kernel) {
        for(float val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

std::vector<std::vector<float>> addPadding(const std::vector<std::vector<float>>& input, int padding) {
    // Original image
    // 1 2 3 4
    // 5 6 7 8
    // 9 0 1 2
    // 3 4 5 6

    // Example: padding = 2
    // 0 0 0 0 0 0 0 0
    // 0 0 0 0 0 0 0 0
    // 0 0 1 2 3 4 0 0
    // 0 0 5 6 7 8 0 0
    // 0 0 9 0 1 2 0 0
    // 0 0 3 4 5 6 0 0
    // 0 0 0 0 0 0 0 0
    // 0 0 0 0 0 0 0 0

    int rows = input.size();
    int cols = input[0].size();
    float default_pad_value = 0.0f;
    std::vector<std::vector<float>> paddedImage(rows + 2 * padding, std::vector<float>(cols + 2 * padding, default_pad_value));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            paddedImage[i + padding][j + padding] = input[i][j];
        }
    }

    return paddedImage;
}

// Assumes the image is square and values are between 0 and 1
void writeImageToPPM(const std::vector<std::vector<float>>& image, const std::string& filename) {
    int dim = image.size(); // Assuming square image for simplicity
    std::ofstream ofs(filename, std::ios_base::out | std::ios_base::binary);
    if (!ofs.is_open()) {
        std::cerr << "Could not open file for writing." << std::endl;
        return;
    }

    // PPM header: P5 means a grayscale image, then specify dimensions and the max value.
    ofs << "P5\n" << dim << " " << dim << "\n255\n";

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            unsigned char pixel = static_cast<unsigned char>(255 * std::max(0.0f, std::min(1.0f, image[i][j])));
            ofs.write(reinterpret_cast<char*>(&pixel), sizeof(pixel));
        }
    }

    ofs.close();
    std::cout << "Image written to " << filename << std::endl;
}

// Function to create a normalized 5x5 Gaussian kernel
std::vector<std::vector<float>> createNormalizedGaussianKernel(float sigma) {
    int size = 5; // Kernel size
    std::vector<std::vector<float>> kernel(size, std::vector<float>(size, 0));
    float sum = 0.0; // For normalization

    // Calculate the kernel values
    int offset = size / 2; // To center the kernel around (0,0)
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            float x = i - offset;
            float y = j - offset;
            float exponent = -(x*x + y*y) / (2 * sigma * sigma);
            kernel[i][j] = exp(exponent) / (2 * M_PI * sigma * sigma);
            sum += kernel[i][j];
        }
    }

    // Normalize the kernel
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            kernel[i][j] /= sum;
        }
    }

    return kernel;
}


std::vector<std::vector<std::vector<float>>> createMultipleKernals(int numKernals, int rows, int cols, const std::string& distributionType, double param1, double param2) {

    std::vector<std::vector<std::vector<float>>> kernals(numKernals, std::vector<std::vector<float>>(rows, std::vector<float>(cols)));
    for (int i = 0; i < numKernals; ++i) {
        kernals[i] = createDistributedKernel(rows, cols, distributionType, param1, param2);
    }

    return kernals;
}


std::vector<std::vector<float>> applyActivationFunction(const std::vector<std::vector<float>>& input, const std::string& activationFunction) {
    int rows = input.size();
    int cols = input[0].size();
    std::vector<std::vector<float>> output(rows, std::vector<float>(cols));

    if (activationFunction == "relu") {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                output[i][j] = std::max(0.0f, input[i][j]);
            }
        }
    } else if (activationFunction == "sigmoid") {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                output[i][j] = 1.0f / (1.0f + exp(-input[i][j]));
            }
        }
    } else if (activationFunction == "tanh") {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                output[i][j] = tanh(input[i][j]);
            }
        }
    } else {
        std::cerr << "Unknown activation function: " << activationFunction << std::endl;
    }

    return output;
}

std::vector<float> createBias(int numKernals, const std::string& distributionType, double param1, double param2) {
    std::vector<float> bias(numKernals);
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()

    if (distributionType == "uniform") {
        std::uniform_real_distribution<> dis(param1, param2);
        for (int i = 0; i < numKernals; ++i) {
            bias[i] = dis(gen);
        }
    } else if (distributionType == "normal") {
        std::normal_distribution<> dis(param1, param2);
        for (int i = 0; i < numKernals; ++i) {
            bias[i] = dis(gen);
        }
    }

    return bias;
}





int main() {
    const std::string imagesFilename = "../mnist_data/mnist_x_merged.bin"; // Path to your MNIST images binary file
    const std::string labelsFilename = "../mnist_data/mnist_y_merged.bin"; // Path to your MNIST labels binary file
    const int numImages = 70000; // Number of images and labels
    const int imageSize = 28; // Size of each image (for MNIST, it's 28x28)
    const int padding = 2; // Padding size for the input image
    const int kernelSize = 5; // Size of the kernel in LeNet
    const int imageSizeWithPadding = imageSize + 2 * padding; // Size of the image with padding
    const unsigned int seed = 42; // Random seed for kernel initialization
    const int inputDepth = 1; // Number of channels in the input image

    std::vector<float> images; // Vector to hold image data
    std::vector<unsigned char> labels; // Vector to hold label data

    // Load the MNIST images and labels from binary files
    loadMNISTBinary(imagesFilename, images, numImages, imageSize * imageSize);
    labels = readMNISTLabels(labelsFilename, numImages);

    // Now, display the first image and label for debugging
    // displayFirstImageAndLabel(images, labels, imageSize);

    // Assuming loadMNISTBinary loads images in a flat format, extract the first image
    std::vector<std::vector<float>> image(imageSize, std::vector<float>(imageSize));
    for (int i = 0; i < imageSize; ++i) {
        for (int j = 0; j < imageSize; ++j) {
            image[i][j] = images[i * imageSize + j];
        }
    }
    
    // Make sure your input is correctly sized, it should be a 28x28 matrix
    if (image.size() != 28 || image[0].size() != 28) {
        std::cerr << "Input image is not correctly sized." << std::endl;
        return 1; // Exit with an error code
    }

    // Create a kernal for the first layer of lenet
    // std::vector<std::vector<float>> kernel = createDistributedKernel(kernelSize, kernelSize, "uniform", 0.0, 1.0);
    // This kernal blurs the image
    // std::vector<std::vector<float>> testKernel = createNormalizedGaussianKernel(1.0);

    // Print the kernel
    // printKernel(kernel);

    // Add padding to the first image
    std::vector<std::vector<float>> paddedImage = addPadding(image, padding);

    // Verify the padded image size
    if (paddedImage.size() != imageSizeWithPadding || paddedImage[0].size() != imageSizeWithPadding) {
        std::cerr << "Padded image is not correctly sized." << std::endl;
        return 1; // Exit with an error code
    }

    // output image to folder for debugging
    // writeImageToPPM(image, "output.ppm");

    // Perform convolution on the padded image
    // std::vector<std::vector<float>> output(imageSize, std::vector<float>(imageSize));
    // convolve2D(paddedImage, output, kernel);

    // output image to folder for debugging
    // writeImageToPPM(output, "output.ppm");

    // Create lenet layer 1 kernals
    std::vector<std::vector<std::vector<float>>> kernalsLayer1 = createMultipleKernals(6, kernelSize, kernelSize, "uniform", 0.0, 1.0);

    // Initialize 1st layer of lenet
    // ConvolutionLayer definition
    // ConvolutionLayer(inputHeight, inputWidth, inputDepth, filterHeight, filterWidth, horizontalStride, verticalStride, numFilters)
    ConvolutionLayer layer1(imageSizeWithPadding, imageSizeWithPadding, 1, kernelSize, kernelSize, 1, 1, 6);

    // Print the kernals
    for (int i = 0; i < 6; ++i) {
        printKernel(kernalsLayer1[i]);
    }

    // Do first layer of lenet
    std::vector<std::vector<std::vector<float>>> OutputLayer1(6, std::vector<std::vector<float>>(imageSize, std::vector<float>(imageSize)));

    for (int i = 0; i < 6; ++i) {
        convolve2D(paddedImage, OutputLayer1[i], kernalsLayer1[i]);
    }

    // Print after convolution

    // Print feature maps
    for (int i = 0; i < 6; ++i) {
        writeImageToPPM(OutputLayer1[i], "preoutput" + std::to_string(i) + ".ppm");
    }


    // bias
    std::vector<float> biasLayer1 = createBias(6, "uniform", 0.0, 1.0);

    // Add bias
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < imageSize; ++j) {
            for (int k = 0; k < imageSize; ++k) {
                OutputLayer1[i][j][k] += biasLayer1[i];
            }
        }
    }

    // Apply activation function
    std::vector<std::vector<std::vector<float>>> OutputLayer1Activated(6, std::vector<std::vector<float>>(imageSize, std::vector<float>(imageSize)));

    for (int i = 0; i < 6; ++i) {
        OutputLayer1Activated[i] = applyActivationFunction(OutputLayer1[i], "relu");
    }

    // Print feature maps
    for (int i = 0; i < 6; ++i) {
        writeImageToPPM(OutputLayer1Activated[i], "output" + std::to_string(i) + ".ppm");
    }

    // 

    return 0;
}
