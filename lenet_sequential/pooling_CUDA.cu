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
naive_pooling_kernel(int n, int c, int h, int w,
                    int r, int s,
                    int out_h, int out_w,
                    int u, int v, int p, int q,
                    float *in, float *out)
{
    // 获取线程在三维网格中的位置
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;

    // 如果线程超出工作范围则退出
    if (x >= out_h * out_w || y >= c || z >= n)
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

    int in_offset = z * c * h * w + y * h * w + pos_ori_h * w + pos_ori_w;
    int in_channel_offset = h * w;

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
                // 计算卷积和
                sum += in[in_offset_tmp + i * w + j];
                in_offset_tmp += in_channel_offset;
            }
        }
    }

    // 计算输出偏移
    int out_offset = z * c * out_h * out_w + y * out_h * out_w + x;
    out[out_offset] = sum/ (r*s);
}

// CPU 端的卷积计算
void pooling_cpu(float *in, float *out, int n, int c, int h, int w, int r, int s, int u, int v, int p, int q, int out_h, int out_w)
{

    for (int n_num = 0; n_num < n; n_num++)
    {
        for (int c_num = 0; c_num < c; c_num++)
        {
            for (int i = 0; i < out_h; i++)
            {
                for (int j = 0; j < out_w; j++)
                {
                    double sum = 0.0;
                    int pos_h = i * u - p;
                    int pos_w = j * v - q;

                    for (int kh_num = 0; kh_num < r; kh_num++)
                    {
                        for (int kwNum = 0; kwNum < s; kwNum++)
                        {
                            int pos_ori_h = pos_h + kh_num;
                            int pos_ori_w = pos_w + kwNum;
                            if (pos_ori_w >= 0 && pos_ori_h >= 0 && pos_ori_w < w && pos_ori_h < h)
                            {
                                sum += (double)(in[n_num * c * h * w + c_num * (w * h) + pos_ori_h * w + pos_ori_w]);
                            }
                        }
                    }
                    out[n_num * c * out_h * out_w + c_num * out_h * out_w + i * out_w + j] = (float)(sum/(r*s));
                }
            }
        }
    }
}

void pooling_gpu(const int n,
                const int c,
                const int h,
                const int w,
                const int r,
                const int s,
                const int u,
                const int v,
                const int p,
                const int q,
                const int out_h, // 输出高
                const int out_w, // 输出宽
                float *in,
                float *out
){
    float *in_device, *out_device;

    cudaMalloc((void **)&in_device, n * c * h * w * sizeof(float));
    cudaMalloc((void **)&out_device, n * c * out_h * out_w * sizeof(float));

    // 将输入数据和卷积核拷贝到 GPU
    cudaMemcpy(in_device, in, n * c * h * w * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(out_device, out, n * c * out_h * out_w * sizeof(float), cudaMemcpyHostToDevice);

    // 定义线程块的大小
    const int blockDim_x = 16;
    const int blockDim_y = 16;

    // 计算线程块和网格的数量
    const int gridDim_x = (out_h * out_w + blockDim_x - 1) / blockDim_x;
    const int gridDim_y = (c + blockDim_y - 1) / blockDim_y;

    // 定义线程块和网

    dim3 blockDim(blockDim_x, blockDim_y);
    dim3 gridDim(gridDim_x, gridDim_y, n);

    // 调用 kernel 函数
    naive_pooling_kernel<<<gridDim, blockDim>>>(n, c, h, w, r, s, out_h, out_w, u, v, p, q, in_device, out_device);
    // 同步
    cudaDeviceSynchronize();

    // 将 GPU 计算的结果拷贝到 CPU
    cudaMemcpy(out, out_device, n * c * out_h * out_w * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(in_device);
    cudaFree(out_device);
}

vector<vector<vector<float>>> averagePooling3D_GPU(
    const vector<vector<vector<float>>> &inputMaps,
    const int poolsize,
    const int stride){

    const int n = 1;                           // batch size
    const int c = inputMaps.size();                           // 通道数
    const int h = inputMaps[0].size();                          // 数据高
    const int w = inputMaps[0][0].size();                          // 数据宽
    const int r = poolsize;                           // 卷积核高
    const int s = poolsize;                           // 卷积核宽
    const int u = stride;                           // 卷积在高方向上的步长
    const int v = stride;                           // 卷积在宽方向上的步长
    const int p = 0;                           // 卷积在高方向上的补边
    const int q = 0;                           // 卷积在宽方向上的补边
    const int out_h = (h + 2 * p - r) / u + 1; // 输出高
    const int out_w = (w + 2 * q - s) / v + 1; // 输出宽

    float *in, *out;
    in = (float *)malloc(n * c * h * w * sizeof(float));
    out = (float *)malloc(n * c * out_h * out_w * sizeof(float));

    int index = 0;
    for (int i = 0; i < c; ++i){
        for (int j=0; j<h;j++){
            for (int k=0; k<w;k++){
                in[index++] = inputMaps[i][j][k];
            }
        }
    }
    pooling_gpu(n,                           // batch size
                c,                          // 通道数
                h,                          // 数据高
                w,                          // 数据宽
                r,                           // 卷积核高
                s,                           // 卷积核宽
                u,                           // 卷积在高方向上的步长
                v,                           // 卷积在宽方向上的步长
                p,                           // 卷积在高方向上的补边
                q,                           // 卷积在宽方向上的补边
                out_h,                       // 输出高
                out_w,                       // 输出宽
                in, out);
    // Initialize output maps
    vector<vector<vector<float>>> outputMaps(c, vector<vector<float>>(out_h, vector<float>(out_w, 0.0)));
    index = 0;
    for (int i = 0; i < k; ++i){
        for (int j=0; j<out_h;j++){
            for (int k=0; k<out_w;k++){
                outputMaps[i][j][k] = out[index++];
            }
        }
    }
    free(in);
    free(out);
    return outputMaps;
}

int main()
{
    // 定义输入数据和卷积核的尺寸
    const int n = 1;                           // batch size
    const int c = 6;                           // 通道数
    const int h = 28;                          // 数据高
    const int w = 28;                          // 数据宽
    const int r = 2;                           // 卷积核高
    const int s = 2;                           // 卷积核宽
    const int u = 2;                           // 卷积在高方向上的步长
    const int v = 2;                           // 卷积在宽方向上的步长
    const int p = 0;                           // 卷积在高方向上的补边
    const int q = 0;                           // 卷积在宽方向上的补边
    const int out_h = (h + 2 * p - r) / u + 1; // 输出高
    const int out_w = (w + 2 * q - s) / v + 1; // 输出宽
    // 分配内存并随机生成输入数据和卷积核
    float *in, *out;
    in = (float *)malloc(n * c * h * w * sizeof(float));
    out = (float *)malloc(n * c * out_h * out_w * sizeof(float));
    // 随机生成输入数据和卷积核
    for (int i = 0; i < n * c * h * w; ++i)
    {
        in[i] = (float)rand() / RAND_MAX;
    }
    pooling_gpu(n,                           // batch size
                c,                          // 通道数
                h,                          // 数据高
                w,                          // 数据宽
                r,                           // 卷积核高
                s,                           // 卷积核宽
                u,                           // 卷积在高方向上的步长
                v,                           // 卷积在宽方向上的步长
                p,                           // 卷积在高方向上的补边
                q,                           // 卷积在宽方向上的补边
                out_h,                       // 输出高
                out_w,                       // 输出宽
                in, out);

    // CPU 端进行卷积计算
    float *out_cpu = (float *)malloc(n * c * out_h * out_w * sizeof(float));
    pooling_cpu(in, out_cpu, n, c, h, w, r, s, u, v, p, q, out_h, out_w);

    // 比较 GPU 和 CPU 计算结果是否一致
    bool pass = true;
    for (int i = 0; i < n * c * out_h * out_w; ++i)
    {
        if (abs(out[i] - out_cpu[i]) > 1e-5)
        {
            pass = false;
            std::cout << "Verification failed at " << i << "!" << std::endl;
            std::cout << "GPU: " << out_cpu[i] << " CPU: " << out[i] << std::endl;
            break;
        }
    }
    std::cout << "Verification Pass"<< std::endl;
    free(in);
    free(out);

    return 0;
}