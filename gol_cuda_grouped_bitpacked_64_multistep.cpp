#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <stdint.h>
#include <inttypes.h>

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
        for (int j = 0; j < ROW_GROUP_SIZE; j++) {
            int ly = threadIdx.y * ROW_GROUP_SIZE + j;

            for (int i = 0; i < COL_GROUP_SIZE; i++) {
                int lx = threadIdx.x * COL_GROUP_SIZE + i;

                // Row pointers for the 3x3 neighborhood centered at row y
                const uint64_t* r0 = shared_in + (ly - 1) * smem_stride; // north
                const uint64_t* r1 = shared_in + (ly + 0) * smem_stride; // center (alive)
                const uint64_t* r2 = shared_in + (ly + 1) * smem_stride; // south

                // Load current word and neighbors (with zero at row ends)
                uint64_t r0L = (lx > 0 && ly > 0)                ? r0[lx - 1] : 0ull;
                uint64_t r0C = (ly > 0)                          ? r0[lx + 0] : 0ull;
                uint64_t r0R = (lx + 1 < smem_stride && ly > 0)  ? r0[lx + 1] : 0ull;

                uint64_t r1L = (lx > 0)                          ? r1[lx - 1] : 0ull;
                uint64_t r1C =                                     r1[lx + 0];
                uint64_t r1R = (lx + 1 < smem_stride) ?            r1[lx + 1] : 0ull;

                uint64_t r2L = (lx > 0 && ly + 1 < n)               ? r2[lx - 1] : 0ull;
                uint64_t r2C = (ly + 1 < n)                         ? r2[lx + 0] : 0ull;
                uint64_t r2R = (lx + 1 < smem_stride && ly + 1 < n) ? r2[lx + 1] : 0ull;

                // Build the 8 neighbor bitboards, aligned to the center bit positions
                uint64_t NW = shl1_carry(r0C, r0L);
                uint64_t  N = r0C;
                uint64_t NE = shr1_carry(r0C, r0R);

                uint64_t  W = shl1_carry(r1C, r1L);
                uint64_t  E = shr1_carry(r1C, r1R);

                uint64_t SW = shl1_carry(r2C, r2L);
                uint64_t  S = r2C;
                uint64_t SE = shr1_carry(r2C, r2R);

                // Sum 8 one-bit operands per bit position using a small adder tree
                // First layer: three groups
                uint64_t sA, cA; add3(NW, N,  NE, sA, cA);
                uint64_t sB, cB; add3(W,  E,  S,  sB, cB);   // (includes S here)
                uint64_t sC, cC; add3(SW, SE, 0ull, sC, cC); // last two + zero

                // Combine the three partial sums -> LSB of total sum (bit0) and new carry c1
                uint64_t bit0, c1; add3(sA, sB, sC, bit0, c1);

                // Sum all carries: c_total = cA + cB + cC + c1
                // Compute csum bits (binary of c_total: up to 4 => 3 bits)
                uint64_t t0, t1;            add3(cA, cB, cC, t0, t1);   // t0 = csum bit0 (pre), t1 = carry
                uint64_t csum0, t2;         add2(t0, c1, csum0, t2);    // csum0 = bit1 of count
                uint64_t csum1, csum2;      add2(t1, t2, csum1, csum2); // csum1 = bit2, csum2 = bit3 of count

                // count bits: [bit3 bit2 bit1 bit0] = [csum2 csum1 csum0 bit0]
                // We need masks for count==2 (0010) and count==3 (0011)
                uint64_t not_bit2 = ~csum1;           // bit2 == 0
                uint64_t eq2 = not_bit2 & csum0 & (~bit0);
                uint64_t eq3 = not_bit2 & csum0 & ( bit0);

                // Alive bitboard (center row)
                uint64_t alive = r1C;

                uint64_t next = eq3 | (alive & eq2);

                // Mask out columns 0 and n-1 (only write 1..n-2), and handle partial last word
                // Build a [start,end) mask in word coordinates
                // int64_t base_col = (int64_t)xw << 6;     // xw*64
                // int64_t start    = std::max<int64_t>(0, 1 - base_col);
                // int64_t end      = std::min<int64_t>(64, (n - 1) - base_col); // exclusive
                // uint64_t valid = 0ull;
                // if (end > start) {
                //     int width = (int)(end - start);
                //     valid = (width == 64) ? ~0ull : (((~0ull) >> (64 - width)) << start);
                // }
                // next &= valid;

                // Write to the interior row y+1
                shared_out[ly * smem_stride + lx] = next;
            }
        }
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