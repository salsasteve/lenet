#include <iostream>
#include <vector>

// 函数用于执行平均池化操作
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

int main() {
    // 输入矩阵
    std::vector<std::vector<float>> input = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16}
    };

    // 池化大小
    int poolSize = 2;
    int stride = 2;

    // 执行平均池化操作
    std::vector<std::vector<float>> output = averagePooling(input, poolSize, stride);

    // 输出结果
    std::cout << "Output after average pooling:" << std::endl;
    for (const auto& row : output) {
        for (float val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
