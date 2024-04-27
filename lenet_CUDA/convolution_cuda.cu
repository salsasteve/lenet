#include <cassert>
#include <cstdlib>
#include <iostream>
#include "weights_and_bias.h"

/ 5 x 5 convolutional mask
#define MASK_DIM 5
#define DEPTH 1
#defince COUNT 6

// Amount the the matrix will hang over the matrix
#define MASK_OFFSET (MASK_DIM / 2)

// Allocate mask in constant memory
__constant__ int mask[6*1*5*5];

// 2D Convolution Kernel
// Takes:
//  matrix: Input matrix
//  result: Convolution result
//  N:      Dimensions of the matrices
__global__ void convolution_2d(int *matrix, int *result, int N, int depth, int count) {
  // Calculate the global thread positions
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Starting index for calculation
  int start_r = row - MASK_OFFSET;
  int start_c = col - MASK_OFFSET;

  // Temp value for accumulating the result
  int temp = 0;

  // Iterate over all the rows
  for (int i = 0; i < MASK_DIM; i++) {
    // Go over each column
    for (int j = 0; j < MASK_DIM; j++) {
      // Range check for rows
      if ((start_r + i) >= 0 && (start_r + i) < N) {
        // Range check for columns
        if ((start_c + j) >= 0 && (start_c + j) < N) {
          // Accumulate result
          temp += matrix[depth * N * N + (start_r + i) * N + (start_c + j)] *
                  mask[count * DEPTH * MASK_DIM * MASK_DIM + depth * MASK_DIM * MASK_DIM + i * MASK_DIM + j];
        }
      }
    }
  }

  // Write back the result
  result[count*N*N + row * N + col] = temp;
}

// Initializes an n x n matrix with random numbers
// Takes:
//  m : Pointer to the matrix
//  n : Dimension of the matrix (square)
void init_matrix(int *m, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      m[n * i + j] = rand() % 100;
    }
  }
}

// Verifies the 2D convolution result on the CPU
// Takes:
//  m:      Original matrix
//  mask:   Convolutional mask
//  result: Result from the GPU
//  N:      Dimensions of the matrix
void verify_result(int *m, int *mask, int *result, int N) {
  // Temp value for accumulating results
  int temp;

  // Intermediate value for more readable code
  int offset_r;
  int offset_c;

  // Go over each row
  for (int c = 0; c < COUNT; c++) {
    for (int d = 0; d < DEPTH; d++) {
      for (int i = 0; i < N; i++) {
        // Go over each column
        for (int j = 0; j < N; j++) {
          // Reset the temp variable
          temp = 0;

          // Go over each mask row
          for (int k = 0; k < MASK_DIM; k++) {
            // Update offset value for row
            offset_r = i - MASK_OFFSET + k;

            // Go over each mask column
            for (int l = 0; l < MASK_DIM; l++) {
              // Update offset value for column
              offset_c = j - MASK_OFFSET + l;
              // Range checks if we are hanging off the matrix
              if (offset_r >= 0 && offset_r < N) {
                if (offset_c >= 0 && offset_c < N) {
                  // Accumulate partial results
                  temp += m[d * N * N + offset_r * N + offset_c] * mask[c * DEPTH * MASK_DIM * MASK_DIM + DEPTH * MASK_DIM * MASK_DIM + k * MASK_DIM + l];
                }
              }
            }
          }
          // Fail if the results don't match
          assert(result[i * N + j] == temp);
        }
      }
    }
  }
}

int main() {
  // Dimensions of the matrix (2 ^ 10 x 2 ^ 10)
  int N = 1 << 10;

  // Size of the matrix (in bytes)
  size_t input_bytes_n = DEPTH * N * N * sizeof(int);
  size_t output_bytes_n = COUNT * N * N * sizeof(int);
  // Allocate the matrix and initialize it
  int *matrix = new int[DEPTH*N * N];
  int *result = new int[COUNT*N * N];
  init_matrix(matrix, N);

  // Size of the mask in bytes
  size_t bytes_m = DEPTH*COUNT*MASK_DIM * MASK_DIM * sizeof(int);

  // Allocate the mask and initialize it
  int *h_mask = new int[MASK_DIM * MASK_DIM];
                                                                                                                                                                                              123,0-1       62%
  init_matrix(h_mask, MASK_DIM);

  // Allocate device memory
  int *d_matrix;
  int *d_result;
  cudaMalloc(&d_matrix, input_bytes_n);
  cudaMalloc(&d_result, output_bytes_n);

  // Copy data to the device
  cudaMemcpy(d_matrix, matrix, bytes_n, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(mask, h_mask, bytes_m);

  // Calculate grid dimensions
  int THREADS = 16;
  int BLOCKS = (N + THREADS - 1) / THREADS;

  // Dimension launch arguments
  dim3 block_dim(THREADS, THREADS);
  dim3 grid_dim(BLOCKS, BLOCKS);
  for (int i = 0; i < DEPTH; i++) {
    for (int j = 0; j < COUNT; j++) {
      // Perform 2D Convolution
      convolution_2d<<<grid_dim, block_dim>>>(d_matrix, d_result, N, i, j);
    }
  }

  // Copy the result back to the CPU
  cudaMemcpy(result, d_result, bytes_n, cudaMemcpyDeviceToHost);

  // Functional test
  verify_result(matrix, h_mask, result, N);

  std::cout << "COMPLETED SUCCESSFULLY!";

  // Free the memory we allocated
  delete[] matrix;
  delete[] result;
  delete[] h_mask;

  cudaFree(d_matrix);
  cudaFree(d_result);

  return 0;
}
