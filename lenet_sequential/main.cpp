#include "MNISTLoader.h"
#include "DebugTools.h"
#include <iostream>
#include <vector>

int main() {
    const std::string imagesFilename = "../mnist_data/mnist_x_merged.bin"; // Path to your MNIST images binary file
    const std::string labelsFilename = "../mnist_data/mnist_y_merged.bin"; // Path to your MNIST labels binary file
    const int numImages = 70000; // Number of images and labels
    const int imageSize = 28 * 28; // Size of each image (for MNIST, it's 28x28)

    std::vector<float> images; // Vector to hold image data
    std::vector<unsigned char> labels; // Vector to hold label data

    // Load the MNIST images and labels from binary files
    loadMNISTBinary(imagesFilename, images, numImages, imageSize);
    labels = readMNISTLabels(labelsFilename, numImages);

    // Now, display the first image and label for debugging
    displayFirstImageAndLabel(images, labels, imageSize);

    

    return 0;
}
