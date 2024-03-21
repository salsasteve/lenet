#include "MNISTLoader.h"
#include "DebugTools.h"
#include "Convolution.h"
#include <iostream>
#include <vector>

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


int main() {
    const std::string imagesFilename = "../mnist_data/mnist_x_merged.bin"; // Path to your MNIST images binary file
    const std::string labelsFilename = "../mnist_data/mnist_y_merged.bin"; // Path to your MNIST labels binary file
    const int numImages = 70000; // Number of images and labels
    const int imageSize = 28; // Size of each image (for MNIST, it's 28x28)
    const int padding = 2; // Padding size for the input image
    const int kernelSize = 5; // Size of the kernel in LeNet
    const int imageSizeWithPadding = imageSize + 2 * padding; // Size of the image with padding

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
    std::vector<float> kernel(kernelSize * kernelSize, 1.0f);

    // Print the kernal
    // for (int i = 0; i < kernelSize; ++i) {
    //     std::cout << static_cast<int>(kernel[i]) << " ";
    //     if ((i + 1) % 5 == 0) std::cout << std::endl; // New line after each 5 pixels
    // }

    // Add padding to the first image
    std::vector<std::vector<float>> paddedImage = addPadding(image, padding);

    // Verify the padded image size
    if (paddedImage.size() != imageSizeWithPadding || paddedImage[0].size() != imageSizeWithPadding) {
        std::cerr << "Padded image is not correctly sized." << std::endl;
        return 1; // Exit with an error code
    }

    return 0;
}
