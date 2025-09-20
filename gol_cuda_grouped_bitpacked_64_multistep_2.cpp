#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <stdint.h>
#include <inttypes.h>

// These are hand-unrolled, you cannot change them
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
    for (int k = 0; k < steps; k++) {

        int ly = threadIdx.y * ROW_GROUP_SIZE;
        int lx = threadIdx.x * COL_GROUP_SIZE;

        // All the rows eneded to evalute 4 rows.
        // NOTE THE OFFSET NAMING
        const uint64_t* r0 = shared_in + (ly - 1) * smem_stride;
        const uint64_t* r1 = shared_in + (ly + 0) * smem_stride;
        const uint64_t* r2 = shared_in + (ly + 1) * smem_stride;
        const uint64_t* r3 = shared_in + (ly + 2) * smem_stride;
        const uint64_t* r4 = shared_in + (ly + 3) * smem_stride;
        const uint64_t* r5 = shared_in + (ly + 4) * smem_stride;

        // For each row, load the 3 words we need from it, and immediately sum them
        #define LOAD_SUM(i) \
        uint64_t s ## i, c ## i, rC ## i; \
        {\
            bool in_row_bounds = (ly + i - 1) >= 0 && (ly + i - 1) < n; \
            uint64_t rL = (lx > 0               && in_row_bounds) ? (r ## i)[lx - 1] : 0ull; \
            rC ## i     = (                        in_row_bounds) ? (r ## i)[lx + 0] : 0ull; \
            uint64_t rR = (lx + 1 < smem_stride && in_row_bounds) ? (r ## i)[lx + 1] : 0ull; \
            add3(shl1_carry(rC ## i, rL), rC ## i, shr1_carry(rC ## i, rR), s ## i, c ## i); \
        }

        LOAD_SUM(0);
        LOAD_SUM(1);
        LOAD_SUM(2);
        LOAD_SUM(3);
        LOAD_SUM(4);
        LOAD_SUM(5);

        // For each triple of rows, do the adder tree and life logic
        #define PROCESS_TRIPLE(i0, i1, i2) \
        uint64_t next ## i1;\
        {\
            /* Combine the three partial sums -> LSB of total sum (bit0) and new carry cn */\
            uint64_t bit0, cn; add3(s ## i0, s ## i1, s ## i2, bit0, cn); \
            /* Sum all carries: c_total = cA + cB + cC + cn */\
            /* Compute csum bits (binary of c_total: up to 4 => 3 bits) */\
            uint64_t t0, t1;            add3(c ## i0, c ## i1, c ## i2, t0, t1);   /* t0 = csum bit0 (pre), t1 = carry */\
            uint64_t csum0, t2;         add2(t0, cn, csum0, t2);    /* csum0 = bit1 of count */\
            uint64_t csum1, csum2;      add2(t1, t2, csum1, csum2); /* csum1 = bit2, csum2 = bit3 of count */\
            /* count bits: [bit3 bit2 bit1 bit0] = [csum2 csum1 csum0 bit0] */\
            /* We need masks for count==4 (0100) and count==3 (0011) */\
            uint64_t eq4 = (~csum2) & csum1 & (~csum0) & (~bit0);\
            uint64_t eq3 = (~csum2) & (~csum1) & csum0 & bit0;\
            /* Alive bitboard (center row) */\
            uint64_t alive = rC ## i1;\
            /* Next state */\
            next ## i1 = eq3 | (alive & eq4);\
        }

        PROCESS_TRIPLE(0, 1, 2);
        PROCESS_TRIPLE(1, 2, 3);
        PROCESS_TRIPLE(2, 3, 4);
        PROCESS_TRIPLE(3, 4, 5);

        // Write out the calculated values (note this undoes the row offset naming)
        shared_out[(ly + 0) * smem_stride + lx] = next1;
        shared_out[(ly + 1) * smem_stride + lx] = next2;
        shared_out[(ly + 2) * smem_stride + lx] = next3;
        shared_out[(ly + 3) * smem_stride + lx] = next4;

        __syncthreads();
        // Double buffer, swap the pointers
        uint64_t* tmp = shared_in;
        shared_in = shared_out;
        shared_out = tmp;
    }

    // Finally, write out shared memory
    for (int j = 0; j < ROW_GROUP_SIZE; j++) {
        int ly = threadIdx.y * ROW_GROUP_SIZE + j;
        int gy = blockIdx.y * block_write_size_row + ly - row_pad;
        // Don't write to the padding
        if (ly < row_pad || ly >= block_size_row - row_pad)
            continue;
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