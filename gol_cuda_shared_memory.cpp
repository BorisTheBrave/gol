#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

template<int BLOCK_X, int BLOCK_Y, int PAD>
__global__ void gol_tiled_kernel_i8(const int8_t* __restrict__ x,
                                    int8_t* __restrict__ output,
                                    long long rowstride, // elements per row
                                    int W, int H) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int gx = blockIdx.x * BLOCK_X + tx + 1; // global x (column)
  const int gy = blockIdx.y * BLOCK_Y + ty + 1; // global y (row)

  // interior only (needs 1-cell halo in all directions)
  if (gx <= 0 || gy <= 0 || gx >= W - 1 || gy >= H - 1) return;

  // +2 for 1-cell halo
  __shared__ unsigned int tile[BLOCK_Y + 2][BLOCK_X + 2 + PAD];

  // central cell
  tile[ty + 1][tx + 1] = x[(long long)gy * rowstride + gx];

  // halos: left/right
  if (tx == 0)             tile[ty + 1][0]            = x[(long long)gy * rowstride + (gx - 1)];
  if (tx == BLOCK_X - 1)   tile[ty + 1][BLOCK_X + 1]  = x[(long long)gy * rowstride + (gx + 1)];

  // halos: top/bottom
  if (ty == 0)             tile[0][tx + 1]            = x[(long long)(gy - 1) * rowstride + gx];
  if (ty == BLOCK_Y - 1)   tile[BLOCK_Y + 1][tx + 1]  = x[(long long)(gy + 1) * rowstride + gx];

  // corners
  if (tx == 0 && ty == 0)
    tile[0][0] = x[(long long)(gy - 1) * rowstride + (gx - 1)];
  if (tx == BLOCK_X - 1 && ty == 0)
    tile[0][BLOCK_X + 1] = x[(long long)(gy - 1) * rowstride + (gx + 1)];
  if (tx == 0 && ty == BLOCK_Y - 1)
    tile[BLOCK_Y + 1][0] = x[(long long)(gy + 1) * rowstride + (gx - 1)];
  if (tx == BLOCK_X - 1 && ty == BLOCK_Y - 1)
    tile[BLOCK_Y + 1][BLOCK_X + 1] = x[(long long)(gy + 1) * rowstride + (gx + 1)];

  __syncthreads();

  // write result
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

void gol(torch::Tensor x, torch::Tensor out) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA tensors");
  TORCH_CHECK(x.scalar_type() == at::kChar, "only int8");
  TORCH_CHECK(x.stride(1) == 1, "colstride must be 1");
  TORCH_CHECK(x.dim() == 2 && out.dim() == 2, "2D tensors");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kChar, "out int8 CUDA");
  TORCH_CHECK(out.size(0) == x.size(0) && out.size(1) == x.size(1), "size match");

  const int H = (int)x.size(0);
  const int W = (int)x.size(1);
  const long long pitch = (long long)x.stride(0);

  constexpr int BLOCK_Y = 4;
  constexpr int BLOCK_X = 64;
  constexpr int PAD = 0;            // try {0,2,4,8}; 2 or 4 often helps on byte stencils

  dim3 block(BLOCK_X, BLOCK_Y);
  dim3 grid((W - 2 + BLOCK_X - 1) / BLOCK_X,
            (H - 2 + BLOCK_Y - 1) / BLOCK_Y);

  auto stream = at::cuda::getCurrentCUDAStream();
  gol_tiled_kernel_i8<BLOCK_X, BLOCK_Y, PAD><<<grid, block, 0, stream>>>(
      x.data_ptr<int8_t>(), out.data_ptr<int8_t>(), pitch, W, H);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "kernel launch failed");
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gol", &gol, "gol (CUDA, int8)");
}
