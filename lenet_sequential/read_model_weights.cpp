#include "read_model_weights.h"
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;
using FourD = vector<vector<vector<vector<float>>>>;
using TwoD = vector<vector<float>>;
using OneD = vector<float>;




TwoD LoadDenseWeights(const std::string& filename,
                      const int dim1_size,
                      const int dim2_size){
    std::ifstream file(filename, std::ios::binary);
    TwoD dense_weights(dim2_size, vector<float>(dim1_size));

    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);

    if (file.read(buffer.data(), size)) {
        float* data = reinterpret_cast<float*>(buffer.data());
        int data_size = size / sizeof(float);
        std::cout << "Data read from file:" << data_size << std::endl;
        int count = 0;
        for (int i = 0; i < dim2_size; ++i) {
            for (int ii=0; ii<dim1_size;++ii){
                dense_weights[i][ii] = data[count];
                count++;
            }
        }
        return dense_weights;
    } else {
        std::cerr << "Error reading file!" << std::endl;
        return dense_weights;
    }

}

FourD LoadConv2DWeights(const std::string& filename,
                        const int dim1_size,
                        const int dim2_size,
                        const int dim3_size,
                        const int dim4_size) {
    std::ifstream file(filename, std::ios::binary);
    FourD conv2d_weights(dim4_size, vector<vector<vector<float>>>(dim3_size, vector<vector<float>>(dim2_size,vector<float>(dim1_size))));

    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);

    if (file.read(buffer.data(), size)) {
        float* data = reinterpret_cast<float*>(buffer.data());
        int data_size = size / sizeof(float);
        std::cout << "Data read from file:" << data_size << std::endl;
        int count = 0;
        for (int i = 0; i < dim1_size; ++i) {
            for (int ii=0; ii<dim2_size;++ii){
                for (int iii=0; iii<dim3_size; ++iii){
                    for (int iiii=0; iiii<dim4_size; ++iiii){
                        conv2d_weights[iiii][iii][i][ii] = data[count];
                        count++;
                    }
                }
            }
        }
        return conv2d_weights;
    } else {
        std::cerr << "Error reading file!" << std::endl;
        return conv2d_weights;
    }

}

vector<float> LoadBias(const std::string& filename,
                       const int dim1_size){
    std::ifstream file(filename, std::ios::binary);
    vector<float> bias(dim1_size);

    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);

    if (file.read(buffer.data(), size)) {
        float* data = reinterpret_cast<float*>(buffer.data());
        int data_size = size / sizeof(float);
        std::cout << "Data read from file:" << data_size << std::endl;
        int count = 0;
        for (int i = 0; i < dim1_size; ++i) {
            bias[i] = data[count];
            count++;
        }
        return bias;
    } else {
        std::cerr << "Error reading file!" << std::endl;
        return bias;
    }
}



