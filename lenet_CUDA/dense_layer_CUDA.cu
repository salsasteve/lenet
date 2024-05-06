#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <math.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

void sgemm_naive_cpu(float *A, float *B, float *bias, float *C, int M, int N, int K)
{
    for (int x = 0; x < M; x++)
    {
        for (int y = 0; y < N; y++)
        {
            float sum = 0.0f;
            for (int i = 0; i < K; i++)
            {
                sum += A[x * K + i] * B[i * N + y];
            }
            C[x * N + y] = tanhf(sum + bias[x * N + y]);
        }
    }
}

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_blocktiling_1d_kernel(float *A, float *B, float *C, int M, int N, int K)
{
    // the output block that we want to compute in this threadblock
    const int c_row = blockIdx.y;
    const int c_col = blockIdx.x;

    // allocate shared memory for the input and output submatrices
    __shared__ float A_shared[BM * BK];
    __shared__ float B_shared[BK * BN];

    // the inner row & col that we're accessing in this thread
    const int thread_row = threadIdx.x / BN;
    const int thread_col = threadIdx.x % BN;

    // advance pointers to the starting positions
    A += c_row * BM * K;
    B += c_col * BN;
    C += c_row * BM * N + c_col * BN;

    // use to avoid out-of-bounds accesses
    int global_m_pos = c_row * BM * K;
    int global_n_pos = c_col * BN;
    const int m_size = M * K;
    const int n_size = N * K;

    const int A_inner_row = threadIdx.x / BK; // warp-level GMEM coalescing
    const int A_inner_col = threadIdx.x % BK;
    const int B_inner_row = threadIdx.x / BN; // warp-level GMEM coalescing
    const int B_inner_col = threadIdx.x % BN;

    // allocate thread-local cache for results in registerfile
    float thread_results[TM] = {0.0};

    // outer loop over block tiles
    for (int bk_idx = 0; bk_idx < K; bk_idx += BK)
    {
        // load the next block of the input matrices into shared memory
        A_shared[A_inner_row * BK + A_inner_col] = (global_m_pos + A_inner_row * K + A_inner_col < m_size) ? A[A_inner_row * K + A_inner_col] : 0.0f;
        B_shared[B_inner_row * BN + B_inner_col] = (global_n_pos + B_inner_row * N + B_inner_col < n_size) ? B[B_inner_row * N + B_inner_col] : 0.0f;

        // wait for all threads to finish loading
        __syncthreads();

        // advance the pointers
        A += BK;
        B += BK * N;
        global_m_pos += BK;
        global_n_pos += BK * N;

        // compute the partial sum
        for (int dot_idx = 0; dot_idx < BK; dot_idx++)
        {
            // we make the dotproduct loop the outside loop, which facilitates
            // reuse of the Bs entry, which we can cache in a tmp var.
            float tmp_b = B_shared[dot_idx * BN + thread_col];
            for (int res_idx = 0; res_idx < TM; res_idx++)
            {
                thread_results[res_idx] += A_shared[(thread_row * TM + res_idx) * BK + dot_idx] * tmp_b;
            }
        }

        // wait for all threads to finish computing
        __syncthreads();
    }

    for (int res_idx = 0; res_idx < TM; res_idx++)
    {
        if (c_row * BM + thread_row * TM + res_idx < M && c_col * BN + thread_col < N)
        {
            C[(thread_row * TM + res_idx) * N + thread_col] = thread_results[res_idx];
        }
    }
}

__global__ void add_kernel(float *x, float *y, int n){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < n) {
        x[tid] = tanhf(x[tid] + y[tid]);
    }
}


void run_sgemm_blocktiling_1d(float *A, float *B, float *bias, float *C, int m, int n, int k)
{
    const int BM = 64;
    const int BN = 64;
    const int BK = 8;
    const int TM = 8;
    dim3 grid_size(CEIL_DIV(n, BN), CEIL_DIV(m, BM));
    dim3 block_size((BM * BN) / TM);
    sgemm_blocktiling_1d_kernel<BM, BN, BK, TM>
        <<<grid_size, block_size>>>(A, B, C, m, n, k);
    const int block = 256;
    const int grid = (n + block) / block;
    add_kernel<<<grid, block>>>(C, bias, n);
}

void randomize_matrix(float *mat, int N)
{
    for (int i = 0; i < N; i++)
    {
        mat[i] = rand() % 100;
    }
}

std::vector<float> dense_GPU(
    std::vector<float> &input,
    std::vector<float> &bias,
    std::vector<std::vector<float>> &weights,
    int numOutputs)
{
    int m = 1;
    int n = numOutputs;
    int k = weights.size();

    // Allocate memory for matrices
    float *A, *B, *C, *bias;
    float *d_A, *d_B, *d_C, *d_bias;

    A = new float[m * k];
    B = new float[k * n];
    C = new float[m * n];
    bias = new float[m * n];
    int index = 0;
    for (int i = 0; i < k; ++i){
        A[index++] = input[i];
    }
    index = 0;
    for (int i = 0; i < k; ++i){
        for (int j=0; j<n;j++){
            B[index++] = weights[i][j];
        }
    }
    index = 0;
    for (int i = 0; i < n; ++i){
        bias[index++]=biases[i];
    }

    // Allocate device memory
    cudaMalloc((void **)&d_A, m * k * sizeof(float));
    cudaMalloc((void **)&d_B, k * n * sizeof(float));
    cudaMalloc((void **)&d_C, m * n * sizeof(float));
    cudaMalloc((void **)&d_bias, m * n * sizeof(float));

    // Copy matrices to device
    cudaMemcpy(d_A, A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, k * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, m * n * sizeof(float), cudaMemcpyHostToDevice);

    run_sgemm_blocktiling_1d(d_A, d_B, d_bias, d_C, m, n, k);

    // Copy result to host
    cudaMemcpy(C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> output(n);
    index = 0;
    for (int i = 0; i < n; ++i){
        outputMaps[i] = out[index++];
    }
    return output;
}

int main()
{
    int m = 1;
    int n = 120;
    int k = 400;

    // Allocate memory for matrices
    float *A, *B, *C, *C_ref, *bias;
    float *d_A, *d_B, *d_C, *d_bias;

    A = new float[m * k];
    B = new float[k * n];
    C = new float[m * n];
    bias = new float[m * n];
    // save reference result
    C_ref = new float[m * n];

    // Initialize matrices
    randomize_matrix(A, m * k);
    randomize_matrix(B, k * n);
    randomize_matrix(bias, m*n);

    // Allocate device memory
    cudaMalloc((void **)&d_A, m * k * sizeof(float));
    cudaMalloc((void **)&d_B, k * n * sizeof(float));
    cudaMalloc((void **)&d_C, m * n * sizeof(float));
    cudaMalloc((void **)&d_bias, m * n * sizeof(float));

    // Copy matrices to device
    cudaMemcpy(d_A, A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, k * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, m * n * sizeof(float), cudaMemcpyHostToDevice);

    run_sgemm_blocktiling_1d(d_A, d_B, d_bias, d_C, m, n, k);

    // Copy result to host
    cudaMemcpy(C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Run reference sgemm
    sgemm_naive_cpu(A, B, bias, C_ref, m, n, k);

    // Verify result
    for (int i = 0; i < m * n; i++)
    {
        if (C[i] != C_ref[i])
        {
            printf("Error: mismatch at index %d, expected %f, got %f\n", i, C_ref[i], C[i]);
            return 1;
        }
    }
    int iter = 100;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < iter; i++){
        run_sgemm_blocktiling_1d(d_A, d_B, d_bias, d_C, m, n, k);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "GPU time: " << 1000 * elapsedTime / iter << "us" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Success!\n");
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_bias);
    free(A);
    free(B);
    free(C);
    free(C_ref);
    free(bias);
    return 0;
}