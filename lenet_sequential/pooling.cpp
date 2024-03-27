#include "pooling.h"
#include <iostream>
#include <vector>

std::vector<std::vector<float>> averagePooling(const std::vector<std::vector<float>>& input,
                                                int poolSize,
                                                int stride) {
    int inputHeight = input.size();
    int inputWidth = input[0].size();
    int outputHeight = (inputHeight - poolSize)/stride +1;
    int outputWidth = (inputWidth - poolSize)/stride + 1;

    std::vector<std::vector<float>> output(outputHeight, std::vector<float>(outputWidth, 0.0));

    for (int i = 0; i < outputHeight; i++) {
        for (int j = 0; j < outputWidth; j++) {
            float sum = 0.0;
            for (int m = i*stride; m < i*stride+poolSize; ++m) {
                for (int n = j*stride; n < j*stride+poolSize; ++n) {
                    sum += input[m][n];
                }
            }
            output[i][j] = sum / (poolSize * poolSize);
        }
    }

    return output;
}

std::vector<std::vector<std::vector<float>>> averagePooling3D(const std::vector<std::vector<std::vector<float>>>& input,
                                                              int poolSize,
                                                              int stride) {
    std::vector<std::vector<std::vector<float>>> output;
    for (const auto& matrix : input) {
        output.push_back(averagePooling(matrix, poolSize, stride));
    }
    return output;
}
 

// test the pooling function

// int main() {
//     std::vector<std::vector<float>> input = {
//         {1, 2, 3, 4},
//         {5, 6, 7, 8},
//         {9, 10, 11, 12},
//         {13, 14, 15, 16}
//     };

//     std::vector<std::vector<float>> output = averagePooling(input, 2, 2);

//     for (const auto& row : output) {
//         for (auto val : row) {
//             std::cout << val << " ";
//         }
//         std::cout << std::endl;
//     }

//     return 0;
// }

//Result
// 3.5 5.5
// 11.5 13.5