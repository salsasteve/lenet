// MNISTLoader.cpp
#include "MNISTLoader.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>

// CONSTANTS
const std::string imagesFilename = "../mnist_data/mnist_x_test.bin"; // Path to your MNIST images binary file
const std::string labelsFilename = "../mnist_data/mnist_y_test.bin"; // Path to your MNIST labels binary file
const int image_size = 28 * 28; // Size of each image in the dataset
const int num_labels = 10000; // Number of labels in the dataset

void loadMNISTBinary(const std::string& filename, std::vector<float>& images, int num_images, int image_size) {
    // Open the binary file
    std::ifstream file(filename, std::ios::binary);

    // Check if the file was opened successfully
    if (!file) {
        std::cerr << "Could not open the file " << filename << std::endl;
        return;
    }

    // Resize the vector to hold the specified number of images
    images.resize(num_images * image_size);

    // Calculate the total size to read based on the number of images and the size of each image
    size_t total_size = num_images * image_size * sizeof(float);

    // Read the specified amount of image data from the file directly into the vector
    file.read(reinterpret_cast<char*>(images.data()), total_size);

    // Check if we read less data than expected
    if (file.gcount() < static_cast<std::streamsize>(total_size)) {
        std::cerr << "Unexpected end of file. Expected to read " << total_size
                  << " bytes, but only read " << file.gcount() << " bytes." << std::endl;
        // Optional: Resize the vector if fewer images were read
        images.resize(file.gcount() / sizeof(float));
    }
}


std::vector<std::vector<std::vector<float>>> convertTo3DVector(const std::vector<float>& images, int num_images, int image_size) {
    int image_side = std::sqrt(image_size);
    std::vector<std::vector<std::vector<float>>> images_3d(num_images, std::vector<std::vector<float>>(image_side, std::vector<float>(image_side)));

    for (int i = 0; i < num_images; ++i) {
        for (int row = 0; row < image_side; ++row) {
            for (int col = 0; col < image_side; ++col) {
                images_3d[i][row][col] = images[i * image_size + row * image_side + col];
            }
        }
    }

    return images_3d;
}


std::vector<float> readMNISTLabels(const std::string& filename, int num_labels) {
    std::vector<float> labels(num_labels);
    std::ifstream file(filename, std::ios::binary);

    if (file.is_open()) {
        file.read(reinterpret_cast<char*>(labels.data()), num_labels);

        // Check for read failure
        if (!file) {
            std::cerr << "Error: Only " << file.gcount() << " could be read from " << filename << std::endl;
            return std::vector<float>(); // Return an empty vector in case of failure
        }

        file.close();
    } else {
        std::cerr << "Unable to open the file: " << filename << std::endl;
        return std::vector<float>(); // Return an empty vector in case of failure
    }

    return labels;
}


std::vector<Image> getMNISTImages(int num_images) {
    std::vector<float> images;
    
    loadMNISTBinary(imagesFilename, images, num_images, image_size);
    return convertTo3DVector(images, num_images, image_size);
}

std::vector<float> getMNISTLabels() {

    return readMNISTLabels(labelsFilename, num_labels);
}

