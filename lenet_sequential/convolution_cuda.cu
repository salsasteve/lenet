#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <math.h>
#include <vector>
using namespace std;
/*
 * @param n: batch size
 * @param c: number of channel
 * @param h: height
 * @param w: width
 * @param k: number of kernel
 * @param r: kernel height
 * @param s: kernel width
 * @param out_h: output height
 * @param out_w: output width
 * @param u: stride vertical
 * @param v: stride horizontal
 * @param p: padding height
 * @param q: padding width
 * @param in: input
 * @param weight: kernel
 * @param out: output
 */
__global__ void
naive_conv2d_kernel(int n, int c, int h, int w,
                    int k, int r, int s,
                    int out_h, int out_w,
                    int u, int v, int p, int q,
                    float *in, float *weight, float *bias, float *out)
{
    // 获取线程在三维网格中的位置
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;

    // 如果线程超出工作范围则退出
    if (x >= out_h * out_w || y >= k || z >= n)
    {
        return;
    }

    // 当前线程处理的数据点在out_h、out_w上的坐标
    int pos_out_h = x / out_w;
    int pos_out_w = x % out_w;

    // 计算输入数据的坐标
    int pos_ori_h = pos_out_h * u - p;
    int pos_ori_w = pos_out_w * v - q;

    float sum = 0.0;

    int in_offset = z * c * h * w + pos_ori_h * w + pos_ori_w;
    int weight_offset = y * c * r * s;
    int in_channel_offset = h * w;
    int weight_channel_offset = r * s;

    // 执行卷积操作
    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < s; j++)
        {
            int pos_real_h = pos_ori_h + i;
            int pos_real_w = pos_ori_w + j;

            // 只处理有效的数据点
            if (pos_real_h >= 0 && pos_real_w >= 0 && pos_real_w < w && pos_real_h < h)
            {
                int in_offset_tmp = in_offset;
                int wei_offset_tmp = weight_offset;
                for (int channel = 0; channel < c; channel++)
                {
                    // 计算卷积和
                    sum += in[in_offset_tmp + i * w + j] * weight[wei_offset_tmp + i * s + j];
                    in_offset_tmp += in_channel_offset;
                    wei_offset_tmp += weight_channel_offset;
                }
            }
        }
    }

    // 计算输出偏移
    int out_offset = z * k * out_h * out_w + y * out_h * out_w + x;
    out[out_offset] = tanhf(sum+bias[y]);
}

// CPU 端的卷积计算
void conv2d_cpu(float *in, float *pwei, float *bias, float *out, int n, int c, int h, int w, int k, int r, int s, int u, int v, int p, int q, int out_h, int out_w)
{

    for (int n_num = 0; n_num < n; n_num++)
    {
        for (int k_num = 0; k_num < k; k_num++)
        {
            for (int i = 0; i < out_h; i++)
            {
                for (int j = 0; j < out_w; j++)
                {
                    double sum = 0.0;
                    int pos_h = i * u - p;
                    int pos_w = j * v - q;

                    for (int c_num = 0; c_num < c; c_num++)
                    {
                        for (int kh_num = 0; kh_num < r; kh_num++)
                        {
                            for (int kwNum = 0; kwNum < s; kwNum++)
                            {
                                int pos_ori_h = pos_h + kh_num;
                                int pos_ori_w = pos_w + kwNum;
                                if (pos_ori_w >= 0 && pos_ori_h >= 0 && pos_ori_w < w && pos_ori_h < h)
                                {
                                    sum += (double)(in[n_num * c * h * w + c_num * (w * h) + pos_ori_h * w + pos_ori_w] * pwei[k_num * r * s * c + c_num * r * s + kh_num * s + kwNum]);
                                }
                            }
                        }
                    }

                    out[n_num * k * out_h * out_w + k_num * out_h * out_w + i * out_w + j] = tanhf((float)sum+bias[k_num]);
                }
            }
        }
    }
}

void conv2d_gpu(const int n,
                const int c,
                const int h,
                const int w,
                const int k,
                const int r,
                const int s,
                const int u,
                const int v,
                const int p,
                const int q,
                const int out_h, // 输出高
                const int out_w, // 输出宽
                float *in,
                float *weight,
                float *bias,
                float *out
){
    float *in_device, *weight_device, *bias_device, *out_device;

    cudaMalloc((void **)&in_device, n * c * h * w * sizeof(float));
    cudaMalloc((void **)&weight_device, k * c * r * s * sizeof(float));
    cudaMalloc((void **)&bias_device, k * sizeof(float));
    cudaMalloc((void **)&out_device, n * k * out_h * out_w * sizeof(float));

    // 将输入数据和卷积核拷贝到 GPU
    cudaMemcpy(in_device, in, n * c * h * w * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(weight_device, weight, k * c * r * s * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bias_device, bias, k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(out_device, out, n * k * out_h * out_w * sizeof(float), cudaMemcpyHostToDevice);

    // 定义线程块的大小
    const int blockDim_x = 16;
    const int blockDim_y = 16;

    // 计算线程块和网格的数量
    const int gridDim_x = (out_h * out_w + blockDim_x - 1) / blockDim_x;
    const int gridDim_y = (k + blockDim_y - 1) / blockDim_y;

    // 定义线程块和网

    dim3 blockDim(blockDim_x, blockDim_y);
    dim3 gridDim(gridDim_x, gridDim_y, n);

    // 调用 kernel 函数
    naive_conv2d_kernel<<<gridDim, blockDim>>>(n, c, h, w, k, r, s, out_h, out_w, u, v, p, q, in_device, weight_device, bias_device, out_device);
    // 同步
    cudaDeviceSynchronize();

    // 将 GPU 计算的结果拷贝到 CPU
    cudaMemcpy(out, out_device, n * k * out_h * out_w * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(in_device);
    cudaFree(weight_device);
    cudaFree(out_device);
}

vector<vector<vector<float>>> convolve2dDeep_GPU(
    const vector<vector<vector<float>>> &inputMaps,
    const vector<vector<vector<vector<float>>>> &kernels,
    const vector<float> &biases,
    const int stride,
    const int padding){

    const int n = 1;                           // batch size
    const int c = inputMaps.size();                           // 通道数
    const int h = inputMaps[0].size();                          // 数据高
    const int w = inputMaps[0][0].size();                          // 数据宽
    const int k = kernels.size();                           // 卷积核数量
    const int r = kernels[0][0].size();                           // 卷积核高
    const int s = kernels[0][0][0].size();                           // 卷积核宽
    const int u = stride;                           // 卷积在高方向上的步长
    const int v = stride;                           // 卷积在宽方向上的步长
    const int p = padding;                           // 卷积在高方向上的补边
    const int q = padding;                           // 卷积在宽方向上的补边
    const int out_h = (h + 2 * p - r) / u + 1; // 输出高
    const int out_w = (w + 2 * q - s) / v + 1; // 输出宽

    float *in, *weight, *out, *bias;
    in = (float *)malloc(n * c * h * w * sizeof(float));
    weight = (float *)malloc(k * c * r * s * sizeof(float));
    bias = (float *)malloc(k *sizeof(float));
    out = (float *)malloc(n * k * out_h * out_w * sizeof(float));

    int index = 0;
    for (int i = 0; i < c; ++i){
        for (int j=0; j<h;j++){
            for (int k=0; k<w;k++){
                in[index++] = inputMaps[i][j][k];
            }
        }
    }
    index = 0;
    for (int i = 0; i < k; ++i){
        for (int j=0; j<c;j++){
            for (int k=0; k<r;k++){
                for (int l=0; l<s;l++){
                    weight[index++] = kernels[i][j][k][l];
                }
            }
        }
    }
    index = 0;
    for (int i = 0; i < k; ++i){
        bias[index++]=biases[i];
    }
    conv2d_gpu(n,                           // batch size
                c,                          // 通道数
                h,                          // 数据高
                w,                          // 数据宽
                k,                           // 卷积核数量
                r,                           // 卷积核高
                s,                           // 卷积核宽
                u,                           // 卷积在高方向上的步长
                v,                           // 卷积在宽方向上的步长
                p,                           // 卷积在高方向上的补边
                q,                           // 卷积在宽方向上的补边
                out_h,                       // 输出高
                out_w,                       // 输出宽
                in, weight, bias, out);
    // Initialize output maps
    vector<vector<vector<float>>> outputMaps(k, vector<vector<float>>(out_h, vector<float>(out_w, 0.0)));

    // Perform convolution on each output feature map
    std::cout << "Biases count: " << biases.size() << endl;
    std::cout << "Kernels count: " << kernels.size() << endl;
    index = 0;
    for (int i = 0; i < k; ++i){
        for (int j=0; j<out_h;j++){
            for (int k=0; k<out_w;k++){
                outputMaps[i][j][k] = out[index++];
            }
        }
    }
    return outputMaps;
}

//
// int main()
// {
//     // 定义输入数据和卷积核的尺寸
//     const int n = 1;                           // batch size
//     const int c = 1;                           // 通道数
//     const int h = 28;                          // 数据高
//     const int w = 28;                          // 数据宽
//     const int k = 6;                           // 卷积核数量
//     const int r = 5;                           // 卷积核高
//     const int s = 5;                           // 卷积核宽
//     const int u = 1;                           // 卷积在高方向上的步长
//     const int v = 1;                           // 卷积在宽方向上的步长
//     const int p = 2;                           // 卷积在高方向上的补边
//     const int q = 2;                           // 卷积在宽方向上的补边
//     const int out_h = (h + 2 * p - r) / u + 1; // 输出高
//     const int out_w = (w + 2 * q - s) / v + 1; // 输出宽
//     // 分配内存并随机生成输入数据和卷积核
//     float *in, *weight, *bias, *out;
//     in = (float *)malloc(n * c * h * w * sizeof(float));
//     weight = (float *)malloc(k * c * r * s * sizeof(float));
//     bias = (float *)malloc(k *sizeof(float));
//     out = (float *)malloc(n * k * out_h * out_w * sizeof(float));
//     // 随机生成输入数据和卷积核
//     for (int i = 0; i < n * c * h * w; ++i)
//     {
//         in[i] = (float)rand() / RAND_MAX;
//     }
//     for (int i = 0; i < k * c * r * s; ++i)
//     {
//         weight[i] = (float)rand() / RAND_MAX;
//     }
//     for (int i = 0; i < k; ++i)
//     {
//         bias[i] = (float)rand() / RAND_MAX;
//     }
//     conv2d_gpu(n,                           // batch size
//                 c,                          // 通道数
//                 h,                          // 数据高
//                 w,                          // 数据宽
//                 k,                           // 卷积核数量
//                 r,                           // 卷积核高
//                 s,                           // 卷积核宽
//                 u,                           // 卷积在高方向上的步长
//                 v,                           // 卷积在宽方向上的步长
//                 p,                           // 卷积在高方向上的补边
//                 q,                           // 卷积在宽方向上的补边
//                 out_h,                       // 输出高
//                 out_w,                       // 输出宽
//                 in, weight, bias, out);
// //     float *in_device, *weight_device, *out_device;
// //
// //     cudaMalloc((void **)&in_device, n * c * h * w * sizeof(float));
// //     cudaMalloc((void **)&weight_device, k * c * r * s * sizeof(float));
// //     cudaMalloc((void **)&out_device, n * k * out_h * out_w * sizeof(float));
// //
// //
// //     // 将输入数据和卷积核拷贝到 GPU
// //     cudaMemcpy(in_device, in, n * c * h * w * sizeof(float), cudaMemcpyHostToDevice);
// //     cudaMemcpy(weight_device, weight, k * c * r * s * sizeof(float), cudaMemcpyHostToDevice);
// //     cudaMemcpy(out_device, out, n * k * out_h * out_w * sizeof(float), cudaMemcpyHostToDevice);
// //
// //     // 定义线程块的大小
// //     const int blockDim_x = 16;
// //     const int blockDim_y = 16;
// //
// //     // 计算线程块和网格的数量
// //     const int gridDim_x = (out_h * out_w + blockDim_x - 1) / blockDim_x;
// //     const int gridDim_y = (k + blockDim_y - 1) / blockDim_y;
// //
// //     // 定义线程块和网
// //
// //     dim3 blockDim(blockDim_x, blockDim_y);
// //     dim3 gridDim(gridDim_x, gridDim_y, n);
// //
// //     // 调用 kernel 函数
// //     naive_conv2d_kernel<<<gridDim, blockDim>>>(n, c, h, w, k, r, s, out_h, out_w, u, v, p, q, in_device, weight_device, out_device);
// //     // 同步
// //     cudaDeviceSynchronize();
// //
// //     // 将 GPU 计算的结果拷贝到 CPU
// //     cudaMemcpy(out, out_device, n * k * out_h * out_w * sizeof(float), cudaMemcpyDeviceToHost);
//
//     // CPU 端进行卷积计算
//     float *out_cpu = (float *)malloc(n * k * out_h * out_w * sizeof(float));
//     conv2d_cpu(in, weight, bias, out_cpu, n, c, h, w, k, r, s, u, v, p, q, out_h, out_w);
//
//     // 比较 GPU 和 CPU 计算结果是否一致
//     bool pass = true;
//     for (int i = 0; i < n * k * out_h * out_w; ++i)
//     {
//         if (abs(out[i] - out_cpu[i]) > 1e-5)
//         {
//             pass = false;
//             std::cout << "Verification failed at " << i << "!" << std::endl;
//             std::cout << "GPU: " << out_cpu[i] << " CPU: " << out[i] << std::endl;
//             break;
//         }
//     }
//     std::cout << "Verification Pass"<< std::endl;
// //     if (pass)
// //     {
// //         std::cout << "Verification passed!" << std::endl;
// //
// //         int iter = 100;
// //         cudaEvent_t start, stop;
// //         cudaEventCreate(&start);
// //         cudaEventCreate(&stop);
// //         cudaEventRecord(start, 0);
// //         for (int i = 0; i < iter; i++)
// //         {
// //             naive_conv2d_kernel<<<gridDim, blockDim>>>(n, c, h, w, k, r, s, out_h, out_w, u, v, p, q, in_device, weight_device, out_device);
// //         }
// //         cudaEventRecord(stop, 0);
// //         cudaEventSynchronize(stop);
// //         float elapsedTime;
// //         cudaEventElapsedTime(&elapsedTime, start, stop);
// //         std::cout << "GPU time: " << 1000 * elapsedTime / iter << "us" << std::endl;
// //         cudaEventDestroy(start);
// //         cudaEventDestroy(stop);
// //     }
//
//     // 释放内存
// //     cudaFree(in_device);
// //     cudaFree(weight_device);
// //     cudaFree(out_device);
//     free(in);
//     free(weight);
//     free(out);
//
//     return 0;
// }