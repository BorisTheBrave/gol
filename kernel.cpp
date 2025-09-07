
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void gol_kernel_i8(const int8_t* __restrict__ x_ptr, int8_t* __restrict__ out_ptr, long rowstride, long n) {
  long x = blockIdx.x * blockDim.x + threadIdx.x;
  long y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= n || y >= n) return;

  int8_t r00 = 0;
  int8_t r01 = 0;
  int8_t r02 = 0;
  int8_t r10 = 0;
  int8_t r11 = 0;
  int8_t r12 = 0;
  int8_t r20 = 0;
  int8_t r21 = 0;
  int8_t r22 = 0;

  if (y > 0) {
    if (x > 0) r00 = x_ptr[y * rowstride + x - rowstride - 1];
    r01 = x_ptr[y * rowstride + x - rowstride];
    if (x < n - 1) r02 = x_ptr[y * rowstride + x - rowstride + 1];
  }
  
  if (x > 0) r10 = x_ptr[y * rowstride + x - 1];
  r11 = x_ptr[y * rowstride + x];
  if (x < n - 1) r12 = x_ptr[y * rowstride + x + 1];

  if (y < n - 1) {
    if (x > 0) r20 = x_ptr[y * rowstride + x + rowstride - 1];
    r21 = x_ptr[y * rowstride + x + rowstride];
    if (x < n - 1) r22 = x_ptr[y * rowstride + x + rowstride + 1];
  }

  int8_t sum = r00 + r01 + r02 + r10 + r12 + r20 + r21 + r22;

  int8_t result = (r11 > 0) ? ((sum == 2) || (sum == 3) ? 1 : 0) : (sum == 3 ? 1 : 0);

  out_ptr[y * rowstride + x] = result;
}

void gol(torch::Tensor x, torch::Tensor out) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA tensors");
  TORCH_CHECK(x.scalar_type() == at::kChar, "only int8");
  TORCH_CHECK(x.stride(1) == 1, "colstride must be 1");
  TORCH_CHECK(x.dim() == 2, "x must be 2D");
  TORCH_CHECK(x.size(0) == x.size(1), "x must be square");
  TORCH_CHECK(out.is_cuda(), "out must be CUDA tensors");
  TORCH_CHECK(out.scalar_type() == at::kChar, "only int8");
  TORCH_CHECK(out.stride(1) == 1, "colstride must be 1");
  TORCH_CHECK(out.dim() == 2, "out must be 2D");
  TORCH_CHECK(out.size(0) == x.size(0), "out must have the same height");
  TORCH_CHECK(out.size(1) == x.size(1), "out must have the same width");

  const long n = x.size(0);
  const int block_size = 16;
  const int blocks  = (n + block_size - 1) / block_size;
  auto stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(blocks, blocks);
  dim3 block(block_size, block_size);

  gol_kernel_i8<<<grid, block, 0, stream>>>(
      x.data_ptr<int8_t>(), out.data_ptr<int8_t>(), x.stride(0), n);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "kernel launch failed");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gol", &gol, "gol (CUDA, int8)");
}