// MNISTLoader.h
#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <vector>
#include <string>

using Image = std::vector<std::vector<float>>;

// Declare the function to load MNIST data from a binary file
void loadMNISTBinary(const std::string& filename, std::vector<float>& images, int num_images, int image_size);

std::vector<std::vector<std::vector<float>>> convertTo3DVector(const std::vector<float>& images, int num_images, int image_size);

std::vector<float> readMNISTLabels(const std::string& filename, int num_labels);

std::vector<Image> getMNISTImages(int num_images);

std::vector<float> getMNISTLabels();


#endif // MNIST_LOADER_H