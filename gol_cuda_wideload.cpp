
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void gol_kernel_i32(const int8_t* __restrict__ x_ptr, int8_t* __restrict__ out_ptr, int64_t rowstride, int64_t n) {
  int64_t x = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  int64_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if (!(y + 2 < n)) return;

  if (!(x < n)) return;

  // Vectorized read
  uint32_t r00 =               *(uint32_t*)&x_ptr[y * rowstride + 0 * rowstride + x + 0];
  uint32_t r01 = (x + 4 < n) ? *(uint32_t*)&x_ptr[y * rowstride + 0 * rowstride + x + 4] : 0;

  uint32_t r10 =               *(uint32_t*)&x_ptr[y * rowstride + 1 * rowstride + x + 0];
  uint32_t r11 = (x + 4 < n) ? *(uint32_t*)&x_ptr[y * rowstride + 1 * rowstride + x + 4] : 0;

  uint32_t r20 =               *(uint32_t*)&x_ptr[y * rowstride + 2 * rowstride + x + 0];
  uint32_t r21 = (x + 4 < n) ? *(uint32_t*)&x_ptr[y * rowstride + 2 * rowstride + x + 4] : 0;

  // As each cell has 8 bits, but only takes value 0 or 1, we can sum
  // over rows without overflow

  uint32_t s0 = r00 + r10 + r20;
  uint32_t s1 = r01 + r11 + r21;

  // Bit shift to sum across columns
  uint32_t s = s0 + (s0 >> 8) + (s0 >> 16) + (s1 << 16) + (s1 << 24);

  // Extract the sums
  uint32_t sum0 = (s >> 0) & 0xFF;
  uint32_t sum1 = (s >> 8) & 0xFF;
  uint32_t sum2 = (s >> 16) & 0xFF;
  uint32_t sum3 = (s >> 24) & 0xFF;

  uint32_t alive0 = (r10 >> 0) & 1;
  uint32_t alive1 = (r10 >> 1) & 1;
  uint32_t alive2 = (r10 >> 2) & 1;
  uint32_t alive3 = (r10 >> 3) & 1;

  uint8_t result0 = (alive0 & (sum0 == 4)) | (sum0 == 3);
  int8_t result1 = (alive1 & (sum1 == 4)) | (sum1 == 3);
  uint8_t result2 = (alive2 & (sum2 == 4)) | (sum2 == 3);
  uint8_t result3 = (alive3 & (sum3 == 4)) | (sum3 == 3);

  // We write out per byte as it's not aligned to 32 bits
                 out_ptr[(y + 1) * rowstride + (x + 1)] = result0;
                 out_ptr[(y + 1) * rowstride + (x + 2)] = result1;
                 out_ptr[(y + 1) * rowstride + (x + 3)] = result2;
  if (x + 4 < n) out_ptr[(y + 1) * rowstride + (x + 4)] = result3;
}

void gol(torch::Tensor x, torch::Tensor out, int block_size_row, int block_size_col) {
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

  TORCH_CHECK(block_size_col % 4 == 0, "block_size_col must be divisible by 4");
  TORCH_CHECK(x.size(0) % 4 == 0, "n must be divisible by 4");

  const long n = x.size(0);
  const int block_size_col32 = block_size_col / 4;
  const int row_blocks  = (n - 2 + block_size_row - 1) / block_size_row;
  const int col_blocks  = (n - 2 + block_size_col - 1) / block_size_col;
  auto stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(col_blocks, row_blocks);
  dim3 block(block_size_col32, block_size_row);

  gol_kernel_i32<<<grid, block, 0, stream>>>(
      x.data_ptr<int8_t>(), out.data_ptr<int8_t>(), x.stride(0), n);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "kernel launch failed");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gol", &gol, "gol (CUDA, int8)");
}
