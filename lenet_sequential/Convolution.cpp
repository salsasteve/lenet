// #include <iostream>
// #include <vector>
#include "Convolution.h"


void convolve2D(const std::vector<std::vector<int>>& input, 
                std::vector<std::vector<int>>& output, 
                const std::vector<std::vector<int>>& kernel) {
    int kernelSize = kernel.size();
    int kCenter = kernelSize / 2;
    int rows = input.size();
    int cols = input[0].size();

    for (int i = 0; i < rows; ++i) {      
        for (int j = 0; j < cols; ++j) {  
            for (int m = 0; m < kernelSize; ++m) {   
                int mm = kernelSize - 1 - m;      
                for (int n = 0; n < kernelSize; ++n) { 
                    int nn = kernelSize - 1 - n; 
                    
                    int ii = i + (m - kCenter);
                    int jj = j + (n - kCenter);

                    if (ii >= 0 && ii < rows && jj >= 0 && jj < cols)
                        output[i][j] += input[ii][jj] * kernel[mm][nn];
                }
            }
        }
    }
}