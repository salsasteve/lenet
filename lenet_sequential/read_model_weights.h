#ifndef READ_MODEL_WEIGHTS_H
#define READ_MODEL_WEIGHTS_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;

using FourD = vector<vector<vector<vector<float>>>>;
using TwoD = vector<vector<float>>;
using OneD = vector<float>;

// Function to load 2D dense weights from a binary file
TwoD LoadDenseWeights(const std::string& filename, const int dim1_size, const int dim2_size);

// Function to load 4D convolutional weights from a binary file
FourD LoadConv2DWeights(const std::string& filename, const int dim1_size, const int dim2_size, const int dim3_size, const int dim4_size);

// Function to load 1D bias weights from a binary file
OneD LoadBias(const std::string& filename, const int dim1_size);

#endif // READ_MODEL_WEIGHTS_H
