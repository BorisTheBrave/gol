
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

template<int BLOCK_X, int BLOCK_Y>
__global__ void gol_tiled_kernel_i8(const int8_t* __restrict__ x,
                                    int8_t* __restrict__ output,
                                    long long rowstride, // elements per row
                                    int W, int H) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int gx = blockIdx.x * BLOCK_X + tx; // global x (column)
  const int gy = blockIdx.y * BLOCK_Y + ty; // global y (row)

  // +2 for 1-cell halo on each side
  __shared__ unsigned char tile[BLOCK_Y + 2][BLOCK_X + 2];

  // central cell
  tile[ty + 1][tx + 1] = (gx < W && gy < H) ? x[(long long)gy * rowstride + gx] : 0;

  // halos: left/right
  if (tx == 0)       tile[ty + 1][0]            = (gx > 0      && gy < H) ? x[(long long)gy * rowstride + (gx - 1)] : 0;
  if (tx == BLOCK_X - 1)
                     tile[ty + 1][BLOCK_X + 1]  = (gx + 1 < W  && gy < H) ? x[(long long)gy * rowstride + (gx + 1)] : 0;

  // halos: top/bottom
  if (ty == 0)       tile[0][tx + 1]            = (gy > 0      && gx < W) ? x[(long long)(gy - 1) * rowstride + gx] : 0;
  if (ty == BLOCK_Y - 1)
                     tile[BLOCK_Y + 1][tx + 1]  = (gy + 1 < H  && gx < W) ? x[(long long)(gy + 1) * rowstride + gx] : 0;

  // corners
  if (tx == 0 && ty == 0)
    tile[0][0] = (gx > 0 && gy > 0) ? x[(long long)(gy - 1) * rowstride + (gx - 1)] : 0;

  if (tx == BLOCK_X - 1 && ty == 0)
    tile[0][BLOCK_X + 1] = (gx + 1 < W && gy > 0) ? x[(long long)(gy - 1) * rowstride + (gx + 1)] : 0;

  if (tx == 0 && ty == BLOCK_Y - 1)
    tile[BLOCK_Y + 1][0] = (gx > 0 && gy + 1 < H) ? x[(long long)(gy + 1) * rowstride + (gx - 1)] : 0;

  if (tx == BLOCK_X - 1 && ty == BLOCK_Y - 1)
    tile[BLOCK_Y + 1][BLOCK_X + 1] = (gx + 1 < W && gy + 1 < H) ? x[(long long)(gy + 1) * rowstride + (gx + 1)] : 0;

  __syncthreads();

  if (gx < W && gy < H) {
    const int cx = tx + 1, cy = ty + 1;
    int sum =
      tile[cy-1][cx-1] + tile[cy-1][cx] + tile[cy-1][cx+1] +
      tile[cy][cx-1]                 +    tile[cy][cx+1] +
      tile[cy+1][cx-1] + tile[cy+1][cx] + tile[cy+1][cx+1];

    const unsigned char alive = tile[cy][cx] > 0;
    const unsigned char out   = alive ? ((sum == 2 || sum == 3) ? 1 : 0)
                                      : (sum == 3 ? 1 : 0);
    output[(long long)gy * rowstride + gx] = out;
  }
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

  gol_tiled_kernel_i8<16, 16><<<grid, block, 0, stream>>>(
      x.data_ptr<int8_t>(), out.data_ptr<int8_t>(), x.stride(0), n, n);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "kernel launch failed");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gol", &gol, "gol (CUDA, int8)");
}