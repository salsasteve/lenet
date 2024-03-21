// MNISTLoader.h
#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <vector>
#include <string>

// Declare the function to load MNIST data from a binary file
void loadMNISTBinary(const std::string& filename, std::vector<float>& images, int num_images, int image_size);

std::vector<unsigned char> readMNISTLabels(const std::string& filename, int num_labels);


#endif // MNIST_LOADER_H
