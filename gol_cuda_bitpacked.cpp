
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void gol_kernel_bitpacked(const uint8_t* __restrict__ x_ptr, uint8_t* __restrict__ out_ptr, int64_t rowstride, int64_t n) {
  int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if (y >= n - 2) return;
  if (x >= n / 8) return;

  uint8_t r00 =                   x_ptr[y * rowstride + 0 * rowstride + x + 0];
  uint8_t r01 = (x + 1 < n / 8) ? x_ptr[y * rowstride + 0 * rowstride + x + 1] : 0;
  uint8_t r02 = (x + 2 < n / 8) ? x_ptr[y * rowstride + 0 * rowstride + x + 2] : 0;

  uint8_t r10 =                   x_ptr[y * rowstride + 1 * rowstride + x + 0];
  uint8_t r11 = (x + 1 < n / 8) ? x_ptr[y * rowstride + 1 * rowstride + x + 1] : 0;
  uint8_t r12 = (x + 2 < n / 8) ? x_ptr[y * rowstride + 1 * rowstride + x + 2] : 0;

  uint8_t r20 =                   x_ptr[y * rowstride + 2 * rowstride + x + 0];
  uint8_t r21 = (x + 1 < n / 8) ? x_ptr[y * rowstride + 2 * rowstride + x + 1] : 0;
  uint8_t r22 = (x + 2 < n / 8) ? x_ptr[y * rowstride + 2 * rowstride + x + 2] : 0;

  uint8_t v00_7 = (r00 >> 7) & 1;
  uint8_t v01_0 = (r01 >> 0) & 1;
  uint8_t v01_1 = (r01 >> 1) & 1;
  uint8_t v01_2 = (r01 >> 2) & 1;
  uint8_t v01_3 = (r01 >> 3) & 1;
  uint8_t v01_4 = (r01 >> 4) & 1;
  uint8_t v01_5 = (r01 >> 5) & 1;
  uint8_t v01_6 = (r01 >> 6) & 1;
  uint8_t v01_7 = (r01 >> 7) & 1;
  uint8_t v02_0 = (r02 >> 0) & 1;

  uint8_t v10_7 = (r10 >> 7) & 1;
  uint8_t v11_0 = (r11 >> 0) & 1;
  uint8_t v11_1 = (r11 >> 1) & 1;
  uint8_t v11_2 = (r11 >> 2) & 1;
  uint8_t v11_3 = (r11 >> 3) & 1;
  uint8_t v11_4 = (r11 >> 4) & 1;
  uint8_t v11_5 = (r11 >> 5) & 1;
  uint8_t v11_6 = (r11 >> 6) & 1;
  uint8_t v11_7 = (r11 >> 7) & 1;
  uint8_t v12_0 = (r12 >> 0) & 1;

  uint8_t v20_7 = (r20 >> 7) & 1;
  uint8_t v21_0 = (r21 >> 0) & 1;
  uint8_t v21_1 = (r21 >> 1) & 1;
  uint8_t v21_2 = (r21 >> 2) & 1;
  uint8_t v21_3 = (r21 >> 3) & 1;
  uint8_t v21_4 = (r21 >> 4) & 1;
  uint8_t v21_5 = (r21 >> 5) & 1;
  uint8_t v21_6 = (r21 >> 6) & 1;
  uint8_t v21_7 = (r21 >> 7) & 1;
  uint8_t v22_0 = (r22 >> 0) & 1;

  uint8_t sum_0 = v00_7 + v01_0 + v01_1 + v10_7 + v11_1 + v20_7 + v21_0 + v21_1;
  uint8_t sum_1 = v01_0 + v01_1 + v01_2 + v11_0 + v11_2 + v21_0 + v21_1 + v21_2;
  uint8_t sum_2 = v01_1 + v01_2 + v01_3 + v11_1 + v11_3 + v21_1 + v21_2 + v21_3;
  uint8_t sum_3 = v01_2 + v01_3 + v01_4 + v11_2 + v11_4 + v21_2 + v21_3 + v21_4;
  uint8_t sum_4 = v01_3 + v01_4 + v01_5 + v11_3 + v11_5 + v21_3 + v21_4 + v21_5;
  uint8_t sum_5 = v01_4 + v01_5 + v01_6 + v11_4 + v11_6 + v21_4 + v21_5 + v21_6;
  uint8_t sum_6 = v01_5 + v01_6 + v01_7 + v11_5 + v11_7 + v21_5 + v21_6 + v21_7;
  uint8_t sum_7 = v01_6 + v01_7 + v02_0 + v11_6 + v12_0 + v21_6 + v21_7 + v22_0;

  uint8_t result0 = ((v11_0 == 1) & (sum_0 == 2)) | (sum_0 == 3);
  uint8_t result1 = ((v11_1 == 1) & (sum_1 == 2)) | (sum_1 == 3);
  uint8_t result2 = ((v11_2 == 1) & (sum_2 == 2)) | (sum_2 == 3);
  uint8_t result3 = ((v11_3 == 1) & (sum_3 == 2)) | (sum_3 == 3);
  uint8_t result4 = ((v11_4 == 1) & (sum_4 == 2)) | (sum_4 == 3);
  uint8_t result5 = ((v11_5 == 1) & (sum_5 == 2)) | (sum_5 == 3);
  uint8_t result6 = ((v11_6 == 1) & (sum_6 == 2)) | (sum_6 == 3);
  uint8_t result7 = ((v11_7 == 1) & (sum_7 == 2)) | (sum_7 == 3);

  uint8_t result = result0 | (result1 << 1) | (result2 << 2) | (result3 << 3) | (result4 << 4) | (result5 << 5) | (result6 << 6) | (result7 << 7);

  out_ptr[(y + 1) * rowstride + (x + 1)] = result;
}

void gol(torch::Tensor x, torch::Tensor out, int block_size_row, int block_size_col) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA tensors");
  TORCH_CHECK(x.scalar_type() == at::kByte, "only uint8");
  TORCH_CHECK(x.stride(1) == 1, "colstride must be 1");
  TORCH_CHECK(x.dim() == 2, "x must be 2D");
  TORCH_CHECK(x.size(0) == x.size(1) * 8, "x must be square");
  TORCH_CHECK(out.is_cuda(), "out must be CUDA tensors");
  TORCH_CHECK(out.scalar_type() == at::kByte, "only uint8");
  TORCH_CHECK(out.stride(1) == 1, "colstride must be 1");
  TORCH_CHECK(out.dim() == 2, "out must be 2D");
  TORCH_CHECK(out.size(0) == x.size(0), "out must have the same height");
  TORCH_CHECK(out.size(1) == x.size(1), "out must have the same width");

  const long n = x.size(0);
  const int row_blocks  = (n - 2 + block_size_row - 1) / block_size_row;
  const int col_blocks  = (n / 8 + block_size_col - 1) / block_size_col;
  auto stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(col_blocks, row_blocks);
  dim3 block(block_size_col, block_size_row);

  gol_kernel_bitpacked<<<grid, block, 0, stream>>>(
      x.data_ptr<uint8_t>(), out.data_ptr<uint8_t>(), x.stride(0), n);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "kernel launch failed");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gol", &gol, "gol (CUDA, int8)");
}