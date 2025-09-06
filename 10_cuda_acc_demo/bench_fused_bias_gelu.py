import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import time

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_grad_enabled(False)

mod = load(
    name="fused_bias_gelu_ext",
    sources=["fused_bias_gelu.cu"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",           # 允许更快的近似 math（与 fast_gelu 很搭）
        "-lineinfo",
    ],
    extra_cflags=["-O3", "-std=c++17"],
    verbose=False,
)

def bench_once(fn, iters=200, warmup=50):
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1e3  # ms/iter

# 设一个“模型维度”，比如 Transformer 的隐藏维
B, H = 4096, 4096  # 批量×特征维（这里把它展平成一维来演示）
N = B * H

x = torch.randn(N, device="cuda", dtype=torch.float32)
bias = torch.randn(N, device="cuda", dtype=torch.float32)
y = torch.empty_like(x)

# 方案A：PyTorch 逐步算（x+bias 再 GELU）
def run_pt():
    tmp = x + bias
    out = F.gelu(tmp, approximate="tanh")
    y.copy_(out)  # 为了与定制核同形同地（写回 y），避免编译器消掉工作量

# 方案B：自定义 Fused Kernel
def run_fused():
    mod.fused_bias_gelu(x, bias, y)

# 触发 JIT/编译后基准
t_pt = bench_once(run_pt)
t_fused = bench_once(run_fused)

# 简单正确性检查
with torch.no_grad():
    ref = F.gelu(x + bias, approximate="tanh")
    max_err = (ref - y).abs().max().item()

print(f"PyTorch (add + GELU) : {t_pt:.3f} ms/iter")
print(f"Fused  (one kernel)  : {t_fused:.3f} ms/iter")
print(f"Max abs error        : {max_err:.3e}")
