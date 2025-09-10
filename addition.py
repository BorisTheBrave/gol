# Runs vector addition through torch, triton, and cuda.
# This is like a simplified version of the main code.

# %%
import torch
import triton
import triton.language as tl
import matplotlib.pyplot as plt


# %%
device = torch.device('cuda:0')

# %%

@torch.compile
def torch_add(x: torch.Tensor, y: torch.Tensor):
    return x + y

# %%
@triton.jit
def triton_add_kernel(x_ptr, 
               y_ptr,
               output_ptr,
               n_elements,
               BLOCK_SIZE: tl.constexpr,
               ):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def triton_add(x: torch.Tensor, y: torch.Tensor, BLOCK_SIZE: int = None):
    BLOCK_SIZE = BLOCK_SIZE or 1024
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    triton_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


# %%
import torch
from torch.utils.cpp_extension import load_inline

add_kernel_code = r"""

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void add_kernel_i8(const float* __restrict__ x_ptr, const float* __restrict__ y_ptr, float* __restrict__ out_ptr, long n) {
  long x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= n) return;
  out_ptr[x] = x_ptr[x] + y_ptr[x];
}

void add(torch::Tensor x, torch::Tensor y, torch::Tensor out, int block_size) {
  const long n = x.size(0);
  const int blocks = (n + block_size - 1) / block_size;
  auto stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(blocks);
  dim3 block(block_size);

  add_kernel_i8<<<grid, block, 0, stream>>>(
      x.data_ptr<float>(), y.data_ptr<float>(), out.data_ptr<float>(), n);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "kernel launch failed");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add", &add, "add (CUDA, float)");
}
"""

ext = None
def init_ext():
    global ext
    ext = load_inline(
        name="add_ext",
        cpp_sources="",
        cuda_sources=[add_kernel_code],
        with_cuda=True,
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )

def cuda_add(x: torch.Tensor, y: torch.Tensor, BLOCK_SIZE: int = None):
    BLOCK_SIZE = BLOCK_SIZE or 1024
    if ext is None: init_ext()
    output = torch.empty_like(x)
    ext.add(x, y, output, BLOCK_SIZE)
    return output

# %%
# Key results

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        # x_vals=[(512*32) ** 2],
        x_vals=[2**i for i in range(20, 31, 1)],
        x_log=True,
        line_arg='provider',
        line_vals=['torch', 'triton', 'cuda'],
        line_names=['Torch', 'Triton', 'Cuda'],
        ylabel='ms/element',
        plot_name='add main',
        args={}
    ))


def benchmark(provider='cuda', N=2**30, BLOCK_SIZE=None):
    print(provider, N, )
    # create data
    x = torch.rand(N, device=device)
    y = torch.rand(N, device=device)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_add(x, y), quantiles=quantiles, rep=500)
    elif provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_add(x, y, BLOCK_SIZE=BLOCK_SIZE), quantiles=quantiles, rep=500)
    elif provider == 'cuda':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: cuda_add(x, y, BLOCK_SIZE=BLOCK_SIZE), quantiles=quantiles, rep=500)
    else:
        raise ValueError(f"Invalid provider: {provider}")

    return ms / N

benchmark.run(print_data=True, show_plots=True)
# %%
# BLOCK_SIZE results

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['BLOCK_SIZE'],
        x_vals=[2**i for i in range(5,13)],
        x_log=True,
        line_arg='provider',
        line_vals=['torch', 'triton', 'cuda'],
        line_names=['Torch', 'Triton', 'Cuda'],
        ylabel='ms/element',
        plot_name='add main',
        args={}
    ))
def benchmark(provider='cuda', N=2**30, BLOCK_SIZE=None):
    print(provider, N, BLOCK_SIZE)
    # create data
    x = torch.rand(N, device=device)
    y = torch.rand(N, device=device)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_add(x, y), quantiles=quantiles, rep=500)
    elif provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_add(x, y, BLOCK_SIZE=BLOCK_SIZE), quantiles=quantiles, rep=500)
    elif provider == 'cuda':
        if BLOCK_SIZE >= 2048:
            return float('NaN')
        else:
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: cuda_add(x, y, BLOCK_SIZE=BLOCK_SIZE), quantiles=quantiles, rep=500)
    else:
        raise ValueError(f"Invalid provider: {provider}")

    return ms / N

benchmark.run(print_data=True, show_plots=True)


