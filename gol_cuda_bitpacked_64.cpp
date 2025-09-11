#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// gol_bitpacked.cu
#include <stdint.h>

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

// in/out: bit-packed boards, 1 bit per cell (LSB = smaller column index)
// n: board size (square). We compute output for interior cells only.
// row_words: number of 64-bit words per row = (n + 63) >> 6
extern "C" __global__
void gol_kernel_bitpacked_64(const uint64_t* __restrict__ in,
                          uint64_t* __restrict__ out,
                          int64_t n,
                          int64_t row_words)
{
    int xw = blockIdx.x * blockDim.x + threadIdx.x; // word index in row
    int y  = blockIdx.y * blockDim.y + threadIdx.y; // y in [0..n-3], writes to y+1

    if (xw >= row_words || y >= (int)(n - 2)) return;

    // Row pointers for the 3x3 neighborhood centered at row y+1
    const uint64_t* r0 = in + (y + 0) * row_words; // north
    const uint64_t* r1 = in + (y + 1) * row_words; // center (alive)
    const uint64_t* r2 = in + (y + 2) * row_words; // south

    // Load current word and neighbors (with zero at row ends)
    uint64_t r0L = (xw > 0)              ? r0[xw - 1] : 0ull;
    uint64_t r0C =                          r0[xw];
    uint64_t r0R = (xw + 1 < row_words) ? r0[xw + 1] : 0ull;

    uint64_t r1L = (xw > 0)              ? r1[xw - 1] : 0ull;
    uint64_t r1C =                          r1[xw];
    uint64_t r1R = (xw + 1 < row_words) ? r1[xw + 1] : 0ull;

    uint64_t r2L = (xw > 0)              ? r2[xw - 1] : 0ull;
    uint64_t r2C =                          r2[xw];
    uint64_t r2R = (xw + 1 < row_words) ? r2[xw + 1] : 0ull;

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
    int64_t base_col = (int64_t)xw << 6;     // xw*64
    int64_t start    = std::max<int64_t>(0, 1 - base_col);
    int64_t end      = std::min<int64_t>(64, (n - 1) - base_col); // exclusive
    uint64_t valid = 0ull;
    if (end > start) {
        int width = (int)(end - start);
        valid = (width == 64) ? ~0ull : (((~0ull) >> (64 - width)) << start);
    }
    next &= valid;

    // Write to the interior row y+1
    out[(y + 1) * row_words + xw] = next;
}

void gol(torch::Tensor x, torch::Tensor out, int block_size_row, int block_size_col) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA tensors");
  TORCH_CHECK(x.stride(1) == 1, "colstride must be 1");
  TORCH_CHECK(x.dim() == 2, "x must be 2D");
  TORCH_CHECK(out.is_cuda(), "out must be CUDA tensors");
  TORCH_CHECK(out.stride(1) == 1, "colstride must be 1");
  TORCH_CHECK(out.dim() == 2, "out must be 2D");
  TORCH_CHECK(out.size(0) == x.size(0), "out must have the same height");
  TORCH_CHECK(out.size(1) == x.size(1), "out must have the same width");

  const long n = x.size(0);
  int64_t row_words = (n + 63) >> 6;
  const int row_blocks  = (n - 2 + block_size_row - 1) / block_size_row;
  const int col_blocks  = (row_words + block_size_col - 1) / block_size_col;
  auto stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(col_blocks, row_blocks);
  dim3 block(block_size_col, block_size_row);

  gol_kernel_bitpacked_64<<<grid, block, 0, stream>>>(
      x.data_ptr<uint64_t>(), out.data_ptr<uint64_t>(), n, x.stride(0));
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "kernel launch failed");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gol", &gol, "gol (CUDA, int8)");
}