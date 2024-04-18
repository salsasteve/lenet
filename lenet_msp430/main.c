#include "main.h"
#include <stdint.h>



int16_t main(void)
{
    int16_t result = fp_exp(8, 10, 8);

    matrix inputMatrix = { .numRows = 3, .numCols = 1 };
    matrix resultMatrix = { .numRows = 3, .numCols = 1 };
    
    dtype inputData[3] = {195, -62, 449};  // Fixed-point representation of [0.76, -0.24, 1.76]
    dtype resultData[3] = {0};
    
    inputMatrix.data = inputData;
    resultMatrix.data = resultData;

    softmax(&resultMatrix, &inputMatrix, 8);
    int16_t x = int_to_fp(1, 8);
    x = fp_tanh(x, 8);

    return x;
}


