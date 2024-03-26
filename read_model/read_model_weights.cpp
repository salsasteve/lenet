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
    TwoD dense_weights(dim1_size, vector<float>(dim2_size));

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
    FourD conv2d_weights(dim1_size, vector<vector<vector<float>>>(dim2_size, vector<vector<float>>(dim3_size,vector<float>(dim4_size))));

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
                        conv2d_weights[i][ii][iii][iiii] = data[count];
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


int TestLoadConv2DWeights(FourD& result){
    for (const auto& array3d : result) {
        for (const auto& array2d : array3d) {
            for (const auto& array1d : array2d) {
                for (float value : array1d) {
                    std::cout << value << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    std::cout << "Test Load Conv2D weights success" << "\n";
    return 0;
}

int TestLoadDenseWeights(TwoD& result){

    for (const auto& array1d : result) {
        for (float value : array1d) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "Test Load Dense Weights success" << "\n";
    return 0;
}

int TestLoadBias(OneD& result){
    for (float value : result) {
        std::cout << value << " ";
    }
    std::cout << "Test Load Bias success" << "\n";
    return 0;
}
int main() {
    int dim1_size = 5;
    int dim2_size = 5;
    int dim3_size = 1;
    int dim4_size = 6;
    string filename = "parameters/conv2d_1_weights.bin";
    FourD result = LoadConv2DWeights(filename,dim1_size,dim2_size,dim3_size,dim4_size);
    TestLoadConv2DWeights(result);
    
    int dim5_size = 100;
    int dim6_size = 100;
    string filename_2 = "parameters/dense_1_weights.bin";
    TwoD result_2 = LoadDenseWeights(filename_2, dim5_size, dim6_size);
    TestLoadDenseWeights(result_2);

    string filename_3 = "parameters/conv2d_1_bias.bin";
    OneD result3 = LoadBias(filename_3, 6);
    TestLoadBias(result3);
}
