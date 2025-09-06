#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define WARP_SIZE 32
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// fast GELU (tanh 近似)
__device__ __forceinline__ float fast_gelu(float x) {
    // 0.044715 from Hendrycks & Gimpel
    const float k0 = 0.7978845608028654f;   // sqrt(2/pi)
    const float k1 = 0.044715f;
    float u = k0 * (x + k1 * x * x * x);
    float t = tanhf(u);
    return 0.5f * x * (1.0f + t);
}

__global__ void fused_bias_gelu_f32x4_kernel(const float* __restrict__ x,
                                             const float* __restrict__ bias,
                                             float* __restrict__ y,
                                             int N) {
    // 每线程处理 4 个 float（float4 向量化）
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tid * 4;  // 这组4元素的起点

    if (idx >= N) return;

    // 向量化安全加载：如果可能，走 16B 路；否则退化到标量尾部处理
    float4 vx;
    float4 vb;

    if (idx + 3 < N) {
        vx = *(reinterpret_cast<const float4*>(&x[idx]));
        vb = *(reinterpret_cast<const float4*>(&bias[idx]));

        // 元素级计算（已“手动展开”，等价于 #pragma unroll 4）
        vx.x = fast_gelu(vx.x + vb.x);
        vx.y = fast_gelu(vx.y + vb.y);
        vx.z = fast_gelu(vx.z + vb.z);
        vx.w = fast_gelu(vx.w + vb.w);

        *(reinterpret_cast<float4*>(&y[idx])) = vx;
    } else {
        // 尾部（不满4个元素）安全路径，避免 OOB
        for (int k = 0; k < 4; ++k) {
            int j = idx + k;
            if (j < N) {
                float xv = x[j] + bias[j];
                y[j] = fast_gelu(xv);
            }
        }
    }
}

void fused_bias_gelu_launcher(torch::Tensor x,
                              torch::Tensor bias,
                              torch::Tensor y) {
    TORCH_CHECK(x.is_cuda() && bias.is_cuda() && y.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "only float32 in this demo");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(y.dtype() == torch::kFloat32, "y must be float32");
    TORCH_CHECK(x.numel() == bias.numel() && x.numel() == y.numel(), "size mismatch");

    int N = x.numel();
    int threads = 256;                 // 每 block 256 线程（=8 个 warp），吞吐/占用率都不错
    int vec_elems = 4;                 // 每线程处理 4 元素
    int blocks = ( (N + vec_elems - 1) / vec_elems + threads - 1) / threads;

    fused_bias_gelu_f32x4_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        y.data_ptr<float>(),
        N
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "kernel launch failed: ", cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_bias_gelu", &fused_bias_gelu_launcher, "Fused Bias+GELU (float32, x4)");
}
