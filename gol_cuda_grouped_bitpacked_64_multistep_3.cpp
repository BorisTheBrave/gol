#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <stdint.h>
#include <inttypes.h>

#define INNER_STEPS 2
#define COL_GROUP_SIZE 1
#define ROW_GROUP_SIZE 4

__device__ __forceinline__ void add2(uint64_t a, uint64_t b, uint64_t &s, uint64_t &c) {
    s = a ^ b;
    c = a & b;
}

// 1-bit full adder for three inputs (bitwise, SWAR)
__device__ __forceinline__ void add3(uint64_t a, uint64_t b, uint64_t d, uint64_t &s, uint64_t &c) {
    uint64_t x = a ^ b;
    s = x ^ d;
    c = (a & b) | (a & d) | (b & d);
}

__device__ __forceinline__ uint64_t shl1_carry(uint64_t x, uint64_t left_word) {
    // shift left by 1 with carry-in from the MSB of the word to the left
    return (x << 1) | (left_word >> 63);
}
__device__ __forceinline__ uint64_t shr1_carry(uint64_t x, uint64_t right_word) {
    // shift right by 1 with carry-in from the LSB of the word to the right
    return (x >> 1) | (right_word << 63);
}



__device__ __forceinline__ int cdiv(int a, int b) {
    return (a + b - 1) / b;
}

extern "C" __global__
void gol_kernel_bitpacked_64(const uint64_t* __restrict__ in,
                          uint64_t* __restrict__ out,
                          int64_t n,
                          int64_t in_row_stride,
                          int64_t smem_stride,
                          int64_t steps)
{
    // Recover the dimensions used in the calling method
    int64_t word_n = (n + 63) >> 6;
    int row_pad = steps;
    int word_col_pad = cdiv(steps, 64);

    int block_size_row = blockDim.y * ROW_GROUP_SIZE;
    int word_block_size_col = blockDim.x * COL_GROUP_SIZE;

    int block_write_size_row = block_size_row - 2 * row_pad;
    int word_block_write_size_col = word_block_size_col - 2 * word_col_pad;


    // Shared memory is used in two separate arrays of size block_size_row * word_block_size_col
    extern __shared__ char array[];
    uint64_t* shared_in = reinterpret_cast<uint64_t*>(array);
    uint64_t* shared_out = shared_in + block_size_row * word_block_size_col;

    // First fill in shared memory from in

    for (int j = 0; j < ROW_GROUP_SIZE; j++) {
        int ly = threadIdx.y * ROW_GROUP_SIZE + j;
        int gy = blockIdx.y * block_write_size_row + ly - row_pad;
        for (int i = 0; i < COL_GROUP_SIZE; i++) {
            int lx = threadIdx.x * COL_GROUP_SIZE + i;
            int gxw = blockIdx.x * word_block_write_size_col + lx - word_col_pad;

            if (gy < 0 || gy >= n - 2 || gxw < 0 || gxw >= word_n) {
                shared_in[ly * smem_stride + lx] = 0ull;
            }else{
                shared_in[ly * smem_stride + lx] = in[gy * in_row_stride + gxw];
            }
        }
    }

    __syncthreads();


    // Now run k steps, using shared memory
    for (int k = 0; k < steps; k += INNER_STEPS) {

        static_assert(INNER_STEPS <= 64, "INNER_STEPS must be less than or equal to 64 for bit manipulation to work correctly.");
        // This array is small and always accessed by constant offsets after unrolling
        // so hopefully gets stored in registers
        uint64_t data[ROW_GROUP_SIZE + INNER_STEPS * 2][COL_GROUP_SIZE + 2];

        // Load shared memory into data
        #pragma unroll
        for (int i = 0; i < ROW_GROUP_SIZE + INNER_STEPS * 2; i++) {
            int ly = threadIdx.y * ROW_GROUP_SIZE + i - INNER_STEPS;
            #pragma unroll
            for (int j = 0; j < COL_GROUP_SIZE + 2; j++) {
                int lx = threadIdx.x * COL_GROUP_SIZE + j - 1;
                if (ly < 0 || ly >= block_size_row || lx < 0 || lx >= word_block_size_col) {
                    data[i][j] = 0ull;
                }else{
                    data[i][j] = shared_in[ly * smem_stride + lx];
                }
            }
        }

        // Run multiple inner steps
        // Note that each step needs a slighly smaller range than the last.
        #pragma unroll
        for (int k2 = 0; k2 < INNER_STEPS; k2++) {

            // First we pre-compute the sums of triples
            uint64_t s[ROW_GROUP_SIZE + INNER_STEPS * 2][COL_GROUP_SIZE + 2];
            uint64_t c[ROW_GROUP_SIZE + INNER_STEPS * 2][COL_GROUP_SIZE + 2];

            #pragma unroll
            for (int i = k2; i < ROW_GROUP_SIZE + INNER_STEPS * 2 - k2; i++) {
                int x = k2 == INNER_STEPS - 1 ? 1 : 0;
                #pragma unroll
                for (int j = x; j < COL_GROUP_SIZE + 2 - x; j++) {
                    uint64_t r1L = (j > 0) ?                      data[i + 0][j - 1] : 0ull;
                    uint64_t r1C =                                data[i + 0][j + 0];
                    uint64_t r1R = (j + 1 < COL_GROUP_SIZE + 2) ? data[i + 0][j + 1] : 0ull;
                    uint64_t s1, c1; add3(shl1_carry(r1C, r1L), r1C, shr1_carry(r1C, r1R), s1, c1);
                    s[i][j] = s1;
                    c[i][j] = c1;
                }
            }

            // Then we do the rest, given those
            #pragma unroll
            for (int i = k2 + 1; i < ROW_GROUP_SIZE + INNER_STEPS * 2 - k2 - 1; i++) {
                int x = k2 == INNER_STEPS - 1 ? 1 : 0;
                #pragma unroll
                for (int j = x; j < COL_GROUP_SIZE + 2 - x; j++) {
                    uint64_t s0 = s[i - 1][j];
                    uint64_t c0 = c[i - 1][j];
                    uint64_t s1 = s[i + 0][j];
                    uint64_t c1 = c[i + 0][j];
                    uint64_t s2 = s[i + 1][j];
                    uint64_t c2 = c[i + 1][j];
                    uint64_t bit0, cn; add3(s0, s1, s2, bit0, cn);
                    uint64_t t0, t1; add3(c0, c1, c2, t0, t1);
                    uint64_t csum0, t2; add2(t0, cn, csum0, t2);
                    uint64_t csum1, csum2; add2(t1, t2, csum1, csum2);
                    uint64_t eq4 = (~csum2) & csum1 & (~csum0) & (~bit0);
                    uint64_t eq3 = (~csum2) & (~csum1) & csum0 & bit0;
                    uint64_t alive = data[i + 0][j + 0];
                    data[i][j] = eq3 | (alive & eq4);
                }
            }
        }

        // Copy the output back to shared memory, ready for outer steps
        for (int i = 0; i < ROW_GROUP_SIZE; i++) {
            int ly = threadIdx.y * ROW_GROUP_SIZE + i;
            for (int j = 0; j < COL_GROUP_SIZE ; j++) {
                int lx = threadIdx.x * COL_GROUP_SIZE + j;
                shared_out[ly * smem_stride + lx] = data[i + INNER_STEPS][j + 1];
            }
        }


        __syncthreads();
        // Double buffer, swap the pointers
        uint64_t* tmp = shared_in;
        shared_in = shared_out;
        shared_out = tmp;
    }

    // Finally, write out shared memory
    #pragma unroll
    for (int j = 0; j < ROW_GROUP_SIZE; j++) {
        int ly = threadIdx.y * ROW_GROUP_SIZE + j;
        int gy = blockIdx.y * block_write_size_row + ly - row_pad;
        // Don't write to the padding
        if (ly < row_pad || ly >= block_size_row - row_pad)
            continue;
        #pragma unroll
        for (int i = 0; i < COL_GROUP_SIZE; i++) {
            int lx = threadIdx.x * COL_GROUP_SIZE + i;
            int gxw = blockIdx.x * word_block_write_size_col + lx - word_col_pad;
            // Don't write to the padding
            if (lx < word_col_pad || lx >= word_block_size_col - word_col_pad)
                continue;

            // Don't write out of bounds
            if (gy < 0 || gy >= n || gxw < 0 || gxw >= word_n)
                continue;

            out[gy * in_row_stride + gxw] = shared_in[ly * smem_stride + lx];
        }
    }
}

int cdiv2(int a, int b) {
    return (a + b - 1) / b;
}

void gol(torch::Tensor x, torch::Tensor out, int block_size_row, int block_size_col, int steps) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA tensors");
  TORCH_CHECK(x.stride(1) == 1, "colstride must be 1");
  TORCH_CHECK(x.dim() == 2, "x must be 2D");
  TORCH_CHECK(out.is_cuda(), "out must be CUDA tensors");
  TORCH_CHECK(out.stride(1) == 1, "colstride must be 1");
  TORCH_CHECK(out.dim() == 2, "out must be 2D");
  TORCH_CHECK(out.size(0) == x.size(0), "out must have the same height");
  TORCH_CHECK(out.size(1) == x.size(1), "out must have the same width");

  TORCH_CHECK(block_size_col % (COL_GROUP_SIZE * 64) == 0, "block_size_col must be divisible by COL_GROUP_SIZE * 64");
  TORCH_CHECK(block_size_row % ROW_GROUP_SIZE == 0, "block_size_row must be divisible by ROW_GROUP_SIZE");

  TORCH_CHECK(steps % INNER_STEPS == 0, "steps must be divisible by INNER_STEPS");

  
  // Each threadblock is of reads a block of cells (block_size_row, block_size_col)
  // ie reads int64s (block_size_row, word_block_size_col) into smem where word_block_size_col = block_size_col / 64
  // It runs for STEPS, and writes any fully completed int64s
  // i.e. a rect of size (block_size_row - 2 * STEPS, word_block_size_col - cdiv(STEPS, 64))

  // Thus the threadblocks need to stride by block_size_row - 2 * STEPS in the y direction
  // and by word_block_size_col - cdiv(STEPS, 64) in the x direction to ensure everything is written

  int64_t word_block_size_col = block_size_col >> 6;

  int64_t n = x.size(0);
  int64_t word_n = cdiv2(n, 64);

  int64_t row_pad = steps;
  int64_t word_col_pad = cdiv2(steps, 64);

  int64_t block_write_size_row = block_size_row - 2 * row_pad;
  int64_t word_block_write_size_col = word_block_size_col - 2 * word_col_pad;

  TORCH_CHECK(block_write_size_row > 0, "block_size_col must have space for padding for steps");
  TORCH_CHECK(word_block_write_size_col > 0, "block_size_col/64 must have space for padding for steps");

  

  int64_t row_blocks  = cdiv2(n, block_write_size_row);
  int64_t col_blocks  = cdiv2(word_n, word_block_write_size_col);

  auto stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(col_blocks, row_blocks);
  dim3 block(word_block_size_col / COL_GROUP_SIZE, block_size_row / ROW_GROUP_SIZE);

  size_t shared_mem_size = block_size_row * word_block_size_col * sizeof(uint64_t) * 2;

//   printf("n: %zu\n", n);
//   printf("word_n: %zu\n", word_n);
//   printf("shared_mem_size: %zu\n", shared_mem_size);
//   printf("block_size_row: %d\n", block_size_row);
//   printf("word_block_size_col: %zu\n", word_block_size_col);
//   printf("block_size_col: %d\n", block_size_col);
//   printf("row_blocks: %zu\n", row_blocks);
//   printf("col_blocks: %zu\n", col_blocks);
//   printf("grid: %d, %d\n", grid.x, grid.y);
//   printf("block: %d, %d\n", block.x, block.y);

  gol_kernel_bitpacked_64<<<grid, block, shared_mem_size, stream>>>(
      x.data_ptr<uint64_t>(), out.data_ptr<uint64_t>(), n, x.stride(0), word_block_size_col, steps);

  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("cudaGetLastError: %s\n", cudaGetErrorString(err));
  }
  TORCH_CHECK(err == cudaSuccess, "kernel launch failed");

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gol", &gol, "gol (CUDA, int8)");
}


// NB: On my system, this built with
/*
[1/3] c++ -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=gol10_ext -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1018\" -isystem /root/gol/.venv/lib/python3.11/site-packages/torch/include -isystem /root/gol/.venv/lib/python3.11/site-packages/torch/include/torch/csrc/api/include -isystem /usr/local/cuda/include -isystem /usr/include/python3.11 -fPIC -std=c++17 -c /root/.cache/torch_extensions/py311_cu128/gol10_ext/main.cpp -o main.o 
[2/3] /usr/local/cuda/bin/nvcc --generate-dependencies-with-compile --dependency-output cuda.cuda.o.d -DTORCH_EXTENSION_NAME=gol10_ext -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1018\" -isystem /root/gol/.venv/lib/python3.11/site-packages/torch/include -isystem /root/gol/.venv/lib/python3.11/site-packages/torch/include/torch/csrc/api/include -isystem /usr/local/cuda/include -isystem /usr/include/python3.11 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -gencode=arch=compute_86,code=compute_86 -gencode=arch=compute_86,code=sm_86 --compiler-options '-fPIC' -O3 -std=c++17 -c /root/.cache/torch_extensions/py311_cu128/gol10_ext/cuda.cu -o cuda.cuda.o 
[3/3] c++ main.o cuda.cuda.o -shared -L/root/gol/.venv/lib/python3.11/site-packages/torch/lib -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda -ltorch -ltorch_python -L/usr/local/cuda/lib64 -lcudart -o gol10_ext.so
*/