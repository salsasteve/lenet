// MNISTLoader.cpp
#include "MNISTLoader.h"
#include <fstream>
#include <iostream>

void loadMNISTBinary(const std::string& filename, std::vector<float>& images, int num_images, int image_size) {
    // Open the binary file
    std::ifstream file(filename, std::ios::binary);

    // Check if the file was opened successfully
    if (!file) {
        std::cerr << "Could not open the file " << filename << std::endl;
        return;
    }

    // Move to the end of the file
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();

    // Check if the file size is as expected
    if (size != num_images * image_size * sizeof(float)) {
        std::cerr << "Unexpected file size. Expected " << num_images * image_size * sizeof(float)
                  << " bytes, got " << size << " bytes." << std::endl;
        return;
    }

    // Move back to the beginning of the file
    file.seekg(0, std::ios::beg);

    // Resize the vector to hold all images
    images.resize(num_images * image_size);

    // Read the image data from the file directly into the vector
    file.read(reinterpret_cast<char*>(images.data()), size);
}


std::vector<unsigned char> readMNISTLabels(const std::string& filename, int num_labels) {
    std::vector<unsigned char> labels(num_labels);
    std::ifstream file(filename, std::ios::binary);

    if (file.is_open()) {
        file.read(reinterpret_cast<char*>(labels.data()), num_labels);

        // Check for read failure
        if (!file) {
            std::cerr << "Error: Only " << file.gcount() << " could be read from " << filename << std::endl;
            return std::vector<unsigned char>(); // Return an empty vector in case of failure
        }

        file.close();
    } else {
        std::cerr << "Unable to open the file: " << filename << std::endl;
        return std::vector<unsigned char>(); // Return an empty vector in case of failure
    }

    return labels;
}


