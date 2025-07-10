#include <stdio.h>

#define N 1024
#define RADIUS 1
#define BLOCK_SIZE 128

__global__ void stencil_1d(float *in, float *out, int n) {
    __shared__ float temp[BLOCK_SIZE + 2 * RADIUS];

    int gindex = threadIdx.x + blockIdx.x * blockDim.x;  // 全局索引
    int lindex = threadIdx.x + RADIUS;                   // 共享内存索引

    // 主数据加载
    if (gindex < n)
        temp[lindex] = in[gindex];

    // Halo 数据加载（只有前 RADIUS 个线程去做）
    if (threadIdx.x < RADIUS) {
        // 左侧 halo
        if (gindex >= RADIUS)
            temp[lindex - RADIUS] = in[gindex - RADIUS];
        else
            temp[lindex - RADIUS] = 0.0f;

        // 右侧 halo
        if (gindex + BLOCK_SIZE < n)
            temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
        else
            temp[lindex + BLOCK_SIZE] = 0.0f;
    }

    __syncthreads();

    // 卷积计算
    float result = 0;
    if (gindex < n) {
        for (int offset = -RADIUS; offset <= RADIUS; offset++) {
            result += temp[lindex + offset];
        }
        out[gindex] = result;
    }
}

int main() {
    float *h_in = new float[N];
    float *h_out = new float[N];

    // 初始化输入数据
    for (int i = 0; i < N; i++) {
        h_in[i] = 1.0f;  // 简单输入测试
    }

    // 分配 GPU 内存
    float *d_in, *d_out;
    cudaMalloc((void**)&d_in, N * sizeof(float));
    cudaMalloc((void**)&d_out, N * sizeof(float));

    // 拷贝输入数据到 GPU
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    // 启动 kernel
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    stencil_1d<<<numBlocks, BLOCK_SIZE>>>(d_in, d_out, N);

    // 从 GPU 取回结果
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印部分结果
    printf("前10个输出值:\n");
    for (int i = 0; i < 10; i++) {
        printf("out[%d] = %f\n", i, h_out[i]);
    }

    // 清理内存
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out;

    return 0;
}
