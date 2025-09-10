# %%
import torch
import triton
import triton.language as tl
import matplotlib.pyplot as plt


# %%
device = torch.device('cuda:0')

# %%

# Use torch's built in convolution
# We have to convert to float16 because the convolution doesn't support int8

WEIGHT_F16 = torch.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]])[None, None, :, :].to(torch.float16).to(device)

def gol_torch_conv2d(x: torch.Tensor):
    y = torch.nn.functional.conv2d(x[None, :, :].to(dtype=torch.float16), WEIGHT_F16)[0].to(torch.int8)
    y = torch.nn.functional.pad(y, (1, 1, 1, 1), value=0)
    y = torch.where(x > 0, (y == 2) | (y == 3), (y == 3)).to(torch.int8)
    return y

@torch.compile
def gol_torch_conv2d_compiled(x: torch.Tensor):
    return gol_torch_conv2d(x)

# %%
def gol_torch_conv2d_f16(x: torch.Tensor):
    y = torch.nn.functional.conv2d(x[None, :, :].to(dtype=torch.float16), WEIGHT_F16)[0]
    y = torch.nn.functional.pad(y, (1, 1, 1, 1), value=0)
    y = torch.where(x > 0, (y == 2) | (y == 3), (y == 3)).to(torch.float16)
    return y

@torch.compile
def gol_torch_conv2d_f16_compiled(x: torch.Tensor):
    return gol_torch_conv2d_f16(x)

# %%
# Sum neighbors via slicing
def gol_torch_sum(x: torch.Tensor):
    y = torch.zeros_like(x)
    p = [
        slice(0, -2),
        slice(1, -1),
        slice(2, None),
    ]
    for s1 in p:
        for s2 in p:
            y[1:-1, 1:-1] += x[s1, s2]

    return torch.where(x == 1, (y == 2) | (y == 3), (y == 3)).to(torch.int8)    

@torch.compile
def gol_torch_sum_compiled(x: torch.Tensor):
    return gol_torch_sum(x)

# %%

@triton.jit
def gol_triton_1d_kernel(x_ptr, out_ptr, row_stride: tl.int64, N: tl.int64, BLOCK_SIZE: tl.constexpr):
    col_id = tl.program_id(0)
    row_id = tl.program_id(1)

    offsets0 = col_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets1 = col_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) + 1
    offsets2 = col_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) + 2

    mask0 = (offsets0 >= 0) & (offsets0 < N)
    mask1 = (offsets1 >= 0) & (offsets1 < N)
    mask2 = (offsets2 >= 0) & (offsets2 < N)

    row_ptr = x_ptr + row_id * row_stride

    row00 = tl.load(row_ptr + offsets0 + 0 * row_stride, mask=mask0, other=0)
    row01 = tl.load(row_ptr + offsets1 + 0 * row_stride, mask=mask1, other=0)
    row02 = tl.load(row_ptr + offsets2 + 0 * row_stride, mask=mask2, other=0)
    row10 = tl.load(row_ptr + offsets0 + 1 * row_stride, mask=mask0, other=0)
    row11 = tl.load(row_ptr + offsets1 + 1 * row_stride, mask=mask1, other=0)
    row12 = tl.load(row_ptr + offsets2 + 1 * row_stride, mask=mask2, other=0)
    row20 = tl.load(row_ptr + offsets0 + 2 * row_stride, mask=mask0, other=0)
    row21 = tl.load(row_ptr + offsets1 + 2 * row_stride, mask=mask1, other=0)
    row22 = tl.load(row_ptr + offsets2 + 2 * row_stride, mask=mask2, other=0)

    sum = row00 + row01 + row02 + row10 +  row12 + row20 + row21 + row22

    result = tl.where(row11 > 0, (sum == 2) | (sum == 3), sum == 3).to(tl.int8)

    out_offsets = offsets1
    out_mask = (row_id + 1 < N) & (out_offsets >= 1) & (out_offsets < N - 1)

    tl.store(out_ptr + (row_id + 1) * row_stride + out_offsets, result, mask=out_mask)

def gol_triton_1d(x: torch.Tensor, BLOCK_SIZE: int = None, num_warps: int = None, num_stages: int = None):
    # I don't understand block size tuning yet.
    # There seems to be a significant performance difference between 4096 and 8192.
    BLOCK_SIZE = BLOCK_SIZE or 1024

    output = torch.empty_like(x)

    def grid(meta):
        bs = meta['BLOCK_SIZE']
        return (triton.cdiv(x.shape[1] - 2, bs), x.shape[0] - 2)
    
    gol_triton_1d_kernel[grid](x, output, x.stride(0), x.shape[0], BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_stages=num_stages)

    return output

# %%

@triton.jit
def gol_triton_2d_kernel(x_ptr, out_ptr, row_stride: tl.int64, N: tl.int64, BLOCK_SIZE_ROW: tl.constexpr, BLOCK_SIZE_COL: tl.constexpr):
    row_id = tl.program_id(1)
    col_id = tl.program_id(0)


    col_offsets0 = (col_id * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL) + 0)[None, :]
    col_offsets1 = (col_id * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL) + 1)[None, :]
    col_offsets2 = (col_id * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL) + 2)[None, :]

    row_offsets0 = (row_id * BLOCK_SIZE_ROW + tl.arange(0, BLOCK_SIZE_ROW) + 0)[:, None]
    row_offsets1 = (row_id * BLOCK_SIZE_ROW + tl.arange(0, BLOCK_SIZE_ROW) + 1)[:, None]
    row_offsets2 = (row_id * BLOCK_SIZE_ROW + tl.arange(0, BLOCK_SIZE_ROW) + 2)[:, None]

    col_mask0 = (col_offsets0 >= 0) & (col_offsets0 < N)
    col_mask1 = (col_offsets1 >= 0) & (col_offsets1 < N)
    col_mask2 = (col_offsets2 >= 0) & (col_offsets2 < N)

    row_mask0 = (row_offsets0 >= 0) & (row_offsets0 < N)
    row_mask1 = (row_offsets1 >= 0) & (row_offsets1 < N)
    row_mask2 = (row_offsets2 >= 0) & (row_offsets2 < N)

    row00 = tl.load(x_ptr + row_offsets0 * row_stride + col_offsets0, mask=row_mask0 & col_mask0, other=0)
    row01 = tl.load(x_ptr + row_offsets0 * row_stride + col_offsets1, mask=row_mask0 & col_mask1, other=0)
    row02 = tl.load(x_ptr + row_offsets0 * row_stride + col_offsets2, mask=row_mask0 & col_mask2, other=0)
    row10 = tl.load(x_ptr + row_offsets1 * row_stride + col_offsets0, mask=row_mask1 & col_mask0, other=0)
    row11 = tl.load(x_ptr + row_offsets1 * row_stride + col_offsets1, mask=row_mask1 & col_mask1, other=0)
    row12 = tl.load(x_ptr + row_offsets1 * row_stride + col_offsets2, mask=row_mask1 & col_mask2, other=0)
    row20 = tl.load(x_ptr + row_offsets2 * row_stride + col_offsets0, mask=row_mask2 & col_mask0, other=0)
    row21 = tl.load(x_ptr + row_offsets2 * row_stride + col_offsets1, mask=row_mask2 & col_mask1, other=0)
    row22 = tl.load(x_ptr + row_offsets2 * row_stride + col_offsets2, mask=row_mask2 & col_mask2, other=0)

    sum = row00 + row01 + row02 + row10 +  row12 + row20 + row21 + row22

    result = tl.where(row11 > 0, (sum == 2) | (sum == 3), sum == 3).to(tl.int8)

    tl.store(out_ptr + row_offsets1 * row_stride + col_offsets1, result, mask=row_mask1 & col_mask1)

def gol_triton_2d(x: torch.Tensor, BLOCK_SIZE_ROW: int = None, BLOCK_SIZE_COL: int = None):
    if BLOCK_SIZE_ROW is None and BLOCK_SIZE_COL is None:
        BLOCK_SIZE_ROW = 4
        BLOCK_SIZE_COL = 256

    output = torch.empty_like(x)

    grid = (
        triton.cdiv(x.shape[1] - 2, BLOCK_SIZE_COL),
        triton.cdiv(x.shape[0] - 2, BLOCK_SIZE_ROW),
    )

    gol_triton_2d_kernel[grid](x, output, x.stride(0), x.shape[0], BLOCK_SIZE_ROW=BLOCK_SIZE_ROW, BLOCK_SIZE_COL=BLOCK_SIZE_COL, num_stages=1)

    return output

# %%
def is_notebook() -> bool:
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

# if not is_notebook():
#     x = torch.randint(0, 2, (8096, 8096), device=device, dtype=torch.int8)
#     for i in range(500):
#         output = gol_triton_1d(x)
#     print(output)
#     import sys
#     sys.exit()

# %%
import torch
from torch.utils.cpp_extension import load_inline

ext = None
def init_ext():
    global ext
    ext = load_inline(
        name="gol_ext",
        cpp_sources="",            # no separate C++ binding file
        cuda_sources=[open("kernel.cpp").read()],   # contains both kernel and PYBIND11 module
        with_cuda=True,
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )

def gol_cuda(x: torch.Tensor, BLOCK_SIZE_ROW: int = None, BLOCK_SIZE_COL: int = None):
    if ext is None: init_ext()
    if BLOCK_SIZE_ROW is None and BLOCK_SIZE_COL is None:
        BLOCK_SIZE_ROW = 4
        BLOCK_SIZE_COL = 64
    output = torch.empty_like(x)
    ext.gol(x, output, BLOCK_SIZE_ROW, BLOCK_SIZE_COL)
    return output
# %%
import torch
from torch.utils.cpp_extension import load_inline

ext2 = None

def init_ext2():
    global ext2
    ext2 = load_inline(
        name="gol2_ext",
        cpp_sources="",            # no separate C++ binding file
        cuda_sources=[open("kernel2.cpp").read()],   # contains both kernel and PYBIND11 module
        with_cuda=True,
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )

def gol_cuda_shared_memory(x: torch.Tensor):
    if ext2 is None: init_ext2()
    output = torch.empty_like(x)
    ext2.gol(x, output)
    return output

# %%

@triton.jit
def gol_triton_8bit_1d_kernel(x_ptr, out_ptr, row_stride, N: tl.int64, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    col_id = tl.program_id(1)

    offsets0 = col_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) - 1
    offsets1 = col_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets2 = col_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) + 1

    mask0 = (offsets0 >= 0) & (offsets0 < ((N + 7) // 8))
    mask1 = (offsets1 >= 0) & (offsets1 < ((N + 7) // 8))
    mask2 = (offsets2 >= 0) & (offsets2 < ((N + 7) // 8))

    row_ptr = x_ptr + row_id * row_stride

    if row_id > 0:
        row00 = tl.load(row_ptr + offsets0 - row_stride, mask=mask0, other=0)
        row01 = tl.load(row_ptr + offsets1 - row_stride, mask=mask1, other=0)
        row02 = tl.load(row_ptr + offsets2 - row_stride, mask=mask2, other=0)
    else:
        row00 = tl.zeros((BLOCK_SIZE,), dtype=tl.uint8)
        row01 = tl.zeros((BLOCK_SIZE,), dtype=tl.uint8)
        row02 = tl.zeros((BLOCK_SIZE,), dtype=tl.uint8)
    row10 = tl.load(row_ptr + offsets0, mask=mask0, other=0)
    row11 = tl.load(row_ptr + offsets1, mask=mask1, other=0)
    row12 = tl.load(row_ptr + offsets2, mask=mask2, other=0)
    if row_id < N - 1:
        row20 = tl.load(row_ptr + offsets0 + row_stride, mask=mask0, other=0)
        row21 = tl.load(row_ptr + offsets1 + row_stride, mask=mask1, other=0)
        row22 = tl.load(row_ptr + offsets2 + row_stride, mask=mask2, other=0)
    else:
        row20 = tl.zeros((BLOCK_SIZE,), dtype=tl.uint8)
        row21 = tl.zeros((BLOCK_SIZE,), dtype=tl.uint8)
        row22 = tl.zeros((BLOCK_SIZE,), dtype=tl.uint8)

    # Read out all the bits needed
    v00_7 = (row00 >> 7) & 1
    v01_0 = (row01 >> 0) & 1
    v01_1 = (row01 >> 1) & 1
    v01_2 = (row01 >> 2) & 1
    v01_3 = (row01 >> 3) & 1
    v01_4 = (row01 >> 4) & 1
    v01_5 = (row01 >> 5) & 1
    v01_6 = (row01 >> 6) & 1
    v01_7 = (row01 >> 7) & 1
    v02_0 = (row02 >> 0) & 1

    v10_7 = (row10 >> 7) & 1
    v11_0 = (row11 >> 0) & 1
    v11_1 = (row11 >> 1) & 1
    v11_2 = (row11 >> 2) & 1
    v11_3 = (row11 >> 3) & 1
    v11_4 = (row11 >> 4) & 1
    v11_5 = (row11 >> 5) & 1
    v11_6 = (row11 >> 6) & 1
    v11_7 = (row11 >> 7) & 1
    v12_0 = (row12 >> 0) & 1

    v20_7 = (row20 >> 7) & 1
    v21_0 = (row21 >> 0) & 1
    v21_1 = (row21 >> 1) & 1
    v21_2 = (row21 >> 2) & 1
    v21_3 = (row21 >> 3) & 1
    v21_4 = (row21 >> 4) & 1
    v21_5 = (row21 >> 5) & 1
    v21_6 = (row21 >> 6) & 1
    v21_7 = (row21 >> 7) & 1
    v22_0 = (row22 >> 0) & 1

    sum_0 = v00_7 + v01_0 + v01_1 + v10_7 + v11_0 + v11_1 + v20_7 + v21_1
    sum_1 = v01_0 + v01_1 + v01_2 + v11_0 + v11_2 + v21_0 + v21_1 + v21_2
    sum_2 = v01_1 + v01_2 + v01_3 + v11_1 + v11_3 + v21_1 + v21_2 + v21_3
    sum_3 = v01_2 + v01_3 + v01_4 + v11_2 + v11_4 + v21_2 + v21_3 + v21_4
    sum_4 = v01_3 + v01_4 + v01_5 + v11_3 + v11_5 + v21_3 + v21_4 + v21_5
    sum_5 = v01_4 + v01_5 + v01_6 + v11_4 + v11_6 + v21_4 + v21_5 + v21_6
    sum_6 = v01_5 + v01_6 + v01_7 + v11_5 + v11_7 + v21_5 + v21_6 + v21_7
    sum_7 = v01_6 + v01_7 + v02_0 + v11_6 + v12_0 + v21_6 + v21_7 + v22_0

    result0 = ((v11_0 == 1) & (sum_0 == 2)) | (sum_0 == 3)
    result1 = ((v11_1 == 1) & (sum_1 == 2)) | (sum_1 == 3)
    result2 = ((v11_2 == 1) & (sum_2 == 2)) | (sum_2 == 3)
    result3 = ((v11_3 == 1) & (sum_3 == 2)) | (sum_3 == 3)
    result4 = ((v11_4 == 1) & (sum_4 == 2)) | (sum_4 == 3)
    result5 = ((v11_5 == 1) & (sum_5 == 2)) | (sum_5 == 3)
    result6 = ((v11_6 == 1) & (sum_6 == 2)) | (sum_6 == 3)
    result7 = ((v11_7 == 1) & (sum_7 == 2)) | (sum_7 == 3)


    result = result0 | (result1 << 1) | (result2 << 2) | (result3 << 3) | (result4 << 4) | (result5 << 5) | (result6 << 6) | (result7 << 7)

    out_offsets = col_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_mask = (out_offsets >= 0) & (out_offsets < ((N + 7) // 8))

    tl.store(out_ptr + row_id * row_stride + out_offsets, result, mask=out_mask)

def gol_triton_8bit_1d(x: torch.Tensor, BLOCK_SIZE: int = None):
    # I don't understand block size tuning yet.
    # There seems to be a significant performance difference between 4096 and 8192.
    BLOCK_SIZE = BLOCK_SIZE or 128

    output = torch.empty_like(x)

    def grid(meta):
        bs = meta['BLOCK_SIZE']
        return (x.shape[0], triton.cdiv(x.shape[1], bs))
    
    gol_triton_8bit_1d_kernel[grid](x, output, x.stride(0), x.shape[0], BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    return output

def bit_encode(x: torch.Tensor):

    out = torch.zeros((x.shape[0], triton.cdiv(x.shape[1], 8)), device=x.device, dtype=torch.uint8)

    for i in range(8):
        s = (x[:, i::8] << i)
        out[:, 0:s.shape[1]] += s
    return out

def bit_decode(x: torch.Tensor):
    out = torch.zeros((x.shape[0], x.shape[1] * 8), device=x.device, dtype=torch.int8)
    out[:, 0::8] = (x >> 0) & 1
    out[:, 1::8] = (x >> 1) & 1
    out[:, 2::8] = (x >> 2) & 1
    out[:, 3::8] = (x >> 3) & 1
    out[:, 4::8] = (x >> 4) & 1
    out[:, 5::8] = (x >> 5) & 1
    out[:, 6::8] = (x >> 6) & 1
    out[:, 7::8] = (x >> 7) & 1
    return out

# %%

@triton.jit
def gol_triton_32bit_1d_kernel(x_ptr, out_ptr, row_stride, N: tl.int64, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    col_id = tl.program_id(1)

    offsets0 = col_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) - 1
    offsets1 = col_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets2 = col_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) + 1

    mask0 = (offsets0 >= 0) & (offsets0 < ((N + 31) // 32))
    mask1 = (offsets1 >= 0) & (offsets1 < ((N + 31) // 32))
    mask2 = (offsets2 >= 0) & (offsets2 < ((N + 31) // 32))

    row_ptr = x_ptr + row_id * row_stride

    if row_id > 0:
        row00 = tl.load(row_ptr + offsets0 - row_stride, mask=mask0, other=0)
        row01 = tl.load(row_ptr + offsets1 - row_stride, mask=mask1, other=0)
        row02 = tl.load(row_ptr + offsets2 - row_stride, mask=mask2, other=0)
    else:
        row00 = tl.zeros((BLOCK_SIZE,), dtype=tl.uint32)
        row01 = tl.zeros((BLOCK_SIZE,), dtype=tl.uint32)
        row02 = tl.zeros((BLOCK_SIZE,), dtype=tl.uint32)
    row10 = tl.load(row_ptr + offsets0, mask=mask0, other=0)
    row11 = tl.load(row_ptr + offsets1, mask=mask1, other=0)
    row12 = tl.load(row_ptr + offsets2, mask=mask2, other=0)
    if row_id < N - 1:
        row20 = tl.load(row_ptr + offsets0 + row_stride, mask=mask0, other=0)
        row21 = tl.load(row_ptr + offsets1 + row_stride, mask=mask1, other=0)
        row22 = tl.load(row_ptr + offsets2 + row_stride, mask=mask2, other=0)
    else:
        row20 = tl.zeros((BLOCK_SIZE,), dtype=tl.uint32)
        row21 = tl.zeros((BLOCK_SIZE,), dtype=tl.uint32)
        row22 = tl.zeros((BLOCK_SIZE,), dtype=tl.uint32)

    # Isolate the bits by sixes, and sum across rows
    # # (note bits 30, 31 are excluded as we'd run into overflow issues)
    a0 = ((row01 & 0o0101010101) + (row11 & 0o0101010101) + (row21 & 0o0101010101))
    a1 = ((row01 & 0o0202020202) + (row11 & 0o0202020202) + (row21 & 0o0202020202)) >> 1
    a2 = ((row01 & 0o0404040404) + (row11 & 0o0404040404) + (row21 & 0o0404040404)) >> 2
    a3 = ((row01 & 0o1010101010) + (row11 & 0o1010101010) + (row21 & 0o1010101010)) >> 3
    a4 = ((row01 & 0o2020202020) + (row11 & 0o2020202020) + (row21 & 0o2020202020)) >> 4
    a5 = ((row01 & 0o4040404040) + (row11 & 0o4040404040) + (row21 & 0o4040404040)) >> 5

    # Compute triple sums
    b0 = a0 + a1 + a2
    b1 = a1 + a2 + a3
    b2 = a2 + a3 + a4
    b3 = a3 + a4 + a5
    b4 = a4 + a5 + (a0 >> 6)
    b5 = a5 + (a0 >> 6) + (a1 >> 6)


    # Now read out all the sums
    sum_1 = (a0 + a1) & 0o77 # [0, 1]
    sum0 = (b0 >> 0) & 0o77 # [0, 1, 2]
    sum1 = (b1 >> 0) & 0o77
    sum2 = (b2 >> 0) & 0o77
    sum3 = (b3 >> 0) & 0o77
    sum4 = (b4 >> 0) & 0o77
    sum5 = (b5 >> 0) & 0o77
    sum6 = (b0 >> 6) & 0o77
    sum7 = (b1 >> 6) & 0o77
    sum8 = (b2 >> 6) & 0o77
    sum9 = (b3 >> 6) & 0o77
    sum10 = (b4 >> 6) & 0o77
    sum11 = (b5 >> 6) & 0o77
    sum12 = (b0 >> 12) & 0o77
    sum13 = (b1 >> 12) & 0o77
    sum14 = (b2 >> 12) & 0o77
    sum15 = (b3 >> 12) & 0o77
    sum16 = (b4 >> 12) & 0o77
    sum17 = (b5 >> 12) & 0o77
    sum18 = (b0 >> 18) & 0o77
    sum19 = (b1 >> 18) & 0o77
    sum20 = (b2 >> 18) & 0o77
    sum21 = (b3 >> 18) & 0o77
    sum22 = (b4 >> 18) & 0o77
    sum23 = (b5 >> 18) & 0o77
    sum24 = (b0 >> 24) & 0o77
    sum25 = (b1 >> 24) & 0o77
    sum26 = (b2 >> 24) & 0o77
    sum27 = (b3 >> 24) & 0o77 # [27, 28, 29]
    sum28 = (b4 >> 24) & 0o77 # [28, 29]
    sum29 = (b5 >> 24) & 0o77 # [29]
    sum30 = 0 # []

    # Now add in the bits that are missing
    bit_1 = (row00 >> 31) + (row10 >> 31) + (row20 >> 31)
    bit30 = ((row01 >> 30) & 1) + ((row11 >> 30) & 1) + ((row21 >> 30) & 1)
    bit31 = ((row01 >> 31) & 1) + ((row11 >> 31) & 1) + ((row21 >> 31) & 1)
    bit32 = (row02 & 1) + (row12 & 1) + (row22 & 1)

    sum_1 += bit_1
    sum28 += bit30
    sum29 += bit30 + bit31
    sum30 += bit30 + bit31 + bit32

    # Get the actual aliveness
    alive0 = (row11 >> 0) & 1
    alive1 = (row11 >> 1) & 1
    alive2 = (row11 >> 2) & 1
    alive3 = (row11 >> 3) & 1
    alive4 = (row11 >> 4) & 1
    alive5 = (row11 >> 5) & 1
    alive6 = (row11 >> 6) & 1
    alive7 = (row11 >> 7) & 1
    alive8 = (row11 >> 8) & 1
    alive9 = (row11 >> 9) & 1
    alive10 = (row11 >> 10) & 1
    alive11 = (row11 >> 11) & 1
    alive12 = (row11 >> 12) & 1
    alive13 = (row11 >> 13) & 1
    alive14 = (row11 >> 14) & 1
    alive15 = (row11 >> 15) & 1
    alive16 = (row11 >> 16) & 1
    alive17 = (row11 >> 17) & 1
    alive18 = (row11 >> 18) & 1
    alive19 = (row11 >> 19) & 1
    alive20 = (row11 >> 20) & 1
    alive21 = (row11 >> 21) & 1
    alive22 = (row11 >> 22) & 1
    alive23 = (row11 >> 23) & 1
    alive24 = (row11 >> 24) & 1
    alive25 = (row11 >> 25) & 1
    alive26 = (row11 >> 26) & 1
    alive27 = (row11 >> 27) & 1
    alive28 = (row11 >> 28) & 1
    alive29 = (row11 >> 29) & 1
    alive30 = (row11 >> 30) & 1
    alive31 = (row11 >> 31) & 1

    # Finally, do the gol logic
    # Note our sums include the central cell
    # So it's alive ? sum == 3 || sum == 4 : sum == 3
    # i.e. (alive & (sum == 4)) | sum == 3

    alive0 = (alive0 & (sum_1 == 4)) | (sum_1 == 3)
    alive1 = (alive1 & (sum0 == 4)) | (sum0 == 3)
    alive2 = (alive2 & (sum1 == 4)) | (sum1 == 3)
    alive3 = (alive3 & (sum2 == 4)) | (sum2 == 3)
    alive4 = (alive4 & (sum3 == 4)) | (sum3 == 3)
    alive5 = (alive5 & (sum4 == 4)) | (sum4 == 3)
    alive6 = (alive6 & (sum5 == 4)) | (sum5 == 3)
    alive7 = (alive7 & (sum6 == 4)) | (sum6 == 3)
    alive8 = (alive8 & (sum7 == 4)) | (sum7 == 3)
    alive9 = (alive9 & (sum8 == 4)) | (sum8 == 3)
    alive10 = (alive10 & (sum9 == 4)) | (sum9 == 3)
    alive11 = (alive11 & (sum10 == 4)) | (sum10 == 3)
    alive12 = (alive12 & (sum11 == 4)) | (sum11 == 3)
    alive13 = (alive13 & (sum12 == 4)) | (sum12 == 3)
    alive14 = (alive14 & (sum13 == 4)) | (sum13 == 3)
    alive15 = (alive15 & (sum14 == 4)) | (sum14 == 3)
    alive16 = (alive16 & (sum15 == 4)) | (sum15 == 3)
    alive17 = (alive17 & (sum16 == 4)) | (sum16 == 3)
    alive18 = (alive18 & (sum17 == 4)) | (sum17 == 3)
    alive19 = (alive19 & (sum18 == 4)) | (sum18 == 3)
    alive20 = (alive20 & (sum19 == 4)) | (sum19 == 3)
    alive21 = (alive21 & (sum20 == 4)) | (sum20 == 3)
    alive22 = (alive22 & (sum21 == 4)) | (sum21 == 3)
    alive23 = (alive23 & (sum22 == 4)) | (sum22 == 3)
    alive24 = (alive24 & (sum23 == 4)) | (sum23 == 3)
    alive25 = (alive25 & (sum24 == 4)) | (sum24 == 3)
    alive26 = (alive26 & (sum25 == 4)) | (sum25 == 3)
    alive27 = (alive27 & (sum26 == 4)) | (sum26 == 3)
    alive28 = (alive28 & (sum27 == 4)) | (sum27 == 3)
    alive29 = (alive29 & (sum28 == 4)) | (sum28 == 3)
    alive30 = (alive30 & (sum29 == 4)) | (sum29 == 3)
    alive31 = (alive31 & (sum30 == 4)) | (sum30 == 3)

    # Pack bits
    result = (
        (alive0 << 0) | (alive1 << 1) | (alive2 << 2) | (alive3 << 3) | (alive4 << 4) | (alive5 << 5) | (alive6 << 6) | (alive7 << 7) |
        (alive8 << 8) | (alive9 << 9) | (alive10 << 10) | (alive11 << 11) | (alive12 << 12) | (alive13 << 13) | (alive14 << 14) | (alive15 << 15) | (alive16 << 16) |
        (alive17 << 17) | (alive18 << 18) | (alive19 << 19) | (alive20 << 20) | (alive21 << 21) | (alive22 << 22) | (alive23 << 23) | (alive24 << 24) |
        (alive25 << 25) | (alive26 << 26) | (alive27 << 27) | (alive28 << 28) | (alive29 << 29) | (alive30 << 30) | (alive31 << 31)
    )

    out_offsets = col_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_mask = (out_offsets >= 0) & (out_offsets < ((N + 31) // 32))

    tl.store(out_ptr + row_id * row_stride + out_offsets, result, mask=out_mask)

def gol_triton_32bit_1d(x: torch.Tensor, BLOCK_SIZE: int = None):
    BLOCK_SIZE = BLOCK_SIZE or 256

    output = torch.empty_like(x)

    def grid(meta):
        bs = meta['BLOCK_SIZE']
        return (x.shape[0], triton.cdiv(x.shape[1], bs))
    
    gol_triton_32bit_1d_kernel[grid](x, output, x.stride(0), x.shape[0], BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    return output

def long_encode(x: torch.Tensor):
    assert x.shape[1] % 32 == 0
    assert x.is_contiguous()
    x = bit_encode(x)
    # out = torch.empty(0, dtype=torch.uint32, device=x.device)
    # out.set_(x.untyped_storage(), x, (x.shape[0] , x.shape[1] // 4))
    return x.view(torch.uint32)

def long_decode(x: torch.Tensor):
    assert x.is_contiguous()
    # out = torch.empty(0, dtype=torch.uint8, device=x.device)
    # out.set_(x.untyped_storage(), x, (x.shape[0] , x.shape[1] *4))
    out = x.view(torch.uint8)
    return bit_decode(out)

# %%

@triton.jit
def gol_triton_64bit_1d_kernel(x_ptr, out_ptr, row_stride, N: tl.int64, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    col_id = tl.program_id(1)

    offsets0 = col_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) - 1
    offsets1 = col_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets2 = col_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) + 1

    mask0 = (offsets0 >= 0) & (offsets0 < ((N + 63) // 64))
    mask1 = (offsets1 >= 0) & (offsets1 < ((N + 63) // 64))
    mask2 = (offsets2 >= 0) & (offsets2 < ((N + 63) // 64))

    row_ptr = x_ptr + row_id * row_stride

    if row_id > 0:
        row00 = tl.load(row_ptr + offsets0 - row_stride, mask=mask0, other=0)
        row01 = tl.load(row_ptr + offsets1 - row_stride, mask=mask1, other=0)
        row02 = tl.load(row_ptr + offsets2 - row_stride, mask=mask2, other=0)
    else:
        row00 = tl.zeros((BLOCK_SIZE,), dtype=tl.uint64)
        row01 = tl.zeros((BLOCK_SIZE,), dtype=tl.uint64)
        row02 = tl.zeros((BLOCK_SIZE,), dtype=tl.uint64)
    row10 = tl.load(row_ptr + offsets0, mask=mask0, other=0)
    row11 = tl.load(row_ptr + offsets1, mask=mask1, other=0)
    row12 = tl.load(row_ptr + offsets2, mask=mask2, other=0)
    if row_id < N - 1:
        row20 = tl.load(row_ptr + offsets0 + row_stride, mask=mask0, other=0)
        row21 = tl.load(row_ptr + offsets1 + row_stride, mask=mask1, other=0)
        row22 = tl.load(row_ptr + offsets2 + row_stride, mask=mask2, other=0)
    else:
        row20 = tl.zeros((BLOCK_SIZE,), dtype=tl.uint64)
        row21 = tl.zeros((BLOCK_SIZE,), dtype=tl.uint64)
        row22 = tl.zeros((BLOCK_SIZE,), dtype=tl.uint64)

    # Isolate the bits by sixes, and sum across rows
    # # (note bit 63 is excluded)
    a0 = ((row01 & 0o0101010101010101010101) + (row11 & 0o0101010101010101010101) + (row21 & 0o0101010101010101010101))
    a1 = ((row01 & 0o0202020202020202020202) + (row11 & 0o0202020202020202020202) + (row21 & 0o0202020202020202020202)) >> 1
    a2 = ((row01 & 0o0404040404040404040404) + (row11 & 0o0404040404040404040404) + (row21 & 0o0404040404040404040404)) >> 2
    a3 = ((row01 & 0o0010101010101010101010) + (row11 & 0o0010101010101010101010) + (row21 & 0o0010101010101010101010)) >> 3
    a4 = ((row01 & 0o0020202020202020202020) + (row11 & 0o0020202020202020202020) + (row21 & 0o0020202020202020202020)) >> 4
    a5 = ((row01 & 0o0040404040404040404040) + (row11 & 0o0040404040404040404040) + (row21 & 0o0040404040404040404040)) >> 5

    # Compute triple sums
    b0 = a0 + a1 + a2
    b1 = a1 + a2 + a3
    b2 = a2 + a3 + a4
    b3 = a3 + a4 + a5
    b4 = a4 + a5 + (a0 >> 6)
    b5 = a5 + (a0 >> 6) + (a1 >> 6)


    # Now read out all the sums
    sum_1 = (a0 + a1) & 0o77 # [0, 1]
    sum0 = (b0 >> 0) & 0o77 # [0, 1, 2]
    sum1 = (b1 >> 0) & 0o77
    sum2 = (b2 >> 0) & 0o77
    sum3 = (b3 >> 0) & 0o77
    sum4 = (b4 >> 0) & 0o77
    sum5 = (b5 >> 0) & 0o77
    sum6 = (b0 >> 6) & 0o77
    sum7 = (b1 >> 6) & 0o77
    sum8 = (b2 >> 6) & 0o77
    sum9 = (b3 >> 6) & 0o77
    sum10 = (b4 >> 6) & 0o77
    sum11 = (b5 >> 6) & 0o77
    sum12 = (b0 >> 12) & 0o77
    sum13 = (b1 >> 12) & 0o77
    sum14 = (b2 >> 12) & 0o77
    sum15 = (b3 >> 12) & 0o77
    sum16 = (b4 >> 12) & 0o77
    sum17 = (b5 >> 12) & 0o77
    sum18 = (b0 >> 18) & 0o77
    sum19 = (b1 >> 18) & 0o77
    sum20 = (b2 >> 18) & 0o77
    sum21 = (b3 >> 18) & 0o77
    sum22 = (b4 >> 18) & 0o77
    sum23 = (b5 >> 18) & 0o77
    sum24 = (b0 >> 24) & 0o77
    sum25 = (b1 >> 24) & 0o77
    sum26 = (b2 >> 24) & 0o77
    sum27 = (b3 >> 24) & 0o77
    sum28 = (b4 >> 24) & 0o77
    sum29 = (b5 >> 24) & 0o77
    sum30 = (b0 >> 30) & 0o77
    sum31 = (b1 >> 30) & 0o77
    sum32 = (b2 >> 30) & 0o77
    sum33 = (b3 >> 30) & 0o77
    sum34 = (b4 >> 30) & 0o77
    sum35 = (b5 >> 30) & 0o77
    sum36 = (b0 >> 36) & 0o77
    sum37 = (b1 >> 36) & 0o77
    sum38 = (b2 >> 36) & 0o77
    sum39 = (b3 >> 36) & 0o77
    sum40 = (b4 >> 36) & 0o77
    sum41 = (b5 >> 36) & 0o77
    sum42 = (b0 >> 42) & 0o77
    sum43 = (b1 >> 42) & 0o77
    sum44 = (b2 >> 42) & 0o77
    sum45 = (b3 >> 42) & 0o77
    sum46 = (b4 >> 42) & 0o77
    sum47 = (b5 >> 42) & 0o77
    sum48 = (b0 >> 48) & 0o77
    sum49 = (b1 >> 48) & 0o77
    sum50 = (b2 >> 48) & 0o77
    sum51 = (b3 >> 48) & 0o77
    sum52 = (b4 >> 48) & 0o77
    sum53 = (b5 >> 48) & 0o77
    sum54 = (b0 >> 54) & 0o77
    sum55 = (b1 >> 54) & 0o77
    sum56 = (b2 >> 54) & 0o77
    sum57 = (b3 >> 54) & 0o77
    sum58 = (b4 >> 54) & 0o77
    sum59 = (b5 >> 54) & 0o77
    sum60 = (b0 >> 60) & 0o77
    sum61 = (b1 >> 60) & 0o77 # [61, 62]
    sum62 = (b2 >> 60) & 0o77 # [62]

    # Now add in the bits that are missing
    bit_1 = (row00 >> 31) + (row10 >> 31) + (row20 >> 31)
    bit63 = ((row01 >> 63) & 1) + ((row11 >> 63) & 1) + ((row21 >> 63) & 1)
    bit64 = (row02 & 1) + (row12 & 1) + (row22 & 1)

    sum_1 += bit_1
    sum61 += bit63
    sum62 += bit63 + bit64

    # Get the actual aliveness
    alive0 = (row11 >> 0) & 1
    alive1 = (row11 >> 1) & 1
    alive2 = (row11 >> 2) & 1
    alive3 = (row11 >> 3) & 1
    alive4 = (row11 >> 4) & 1
    alive5 = (row11 >> 5) & 1
    alive6 = (row11 >> 6) & 1
    alive7 = (row11 >> 7) & 1
    alive8 = (row11 >> 8) & 1
    alive9 = (row11 >> 9) & 1
    alive10 = (row11 >> 10) & 1
    alive11 = (row11 >> 11) & 1
    alive12 = (row11 >> 12) & 1
    alive13 = (row11 >> 13) & 1
    alive14 = (row11 >> 14) & 1
    alive15 = (row11 >> 15) & 1
    alive16 = (row11 >> 16) & 1
    alive17 = (row11 >> 17) & 1
    alive18 = (row11 >> 18) & 1
    alive19 = (row11 >> 19) & 1
    alive20 = (row11 >> 20) & 1
    alive21 = (row11 >> 21) & 1
    alive22 = (row11 >> 22) & 1
    alive23 = (row11 >> 23) & 1
    alive24 = (row11 >> 24) & 1
    alive25 = (row11 >> 25) & 1
    alive26 = (row11 >> 26) & 1
    alive27 = (row11 >> 27) & 1
    alive28 = (row11 >> 28) & 1
    alive29 = (row11 >> 29) & 1
    alive30 = (row11 >> 30) & 1
    alive31 = (row11 >> 31) & 1
    alive32 = (row11 >> 32) & 1
    alive33 = (row11 >> 33) & 1
    alive34 = (row11 >> 34) & 1
    alive35 = (row11 >> 35) & 1
    alive36 = (row11 >> 36) & 1
    alive37 = (row11 >> 37) & 1
    alive38 = (row11 >> 38) & 1
    alive39 = (row11 >> 39) & 1
    alive40 = (row11 >> 40) & 1
    alive41 = (row11 >> 41) & 1
    alive42 = (row11 >> 42) & 1
    alive43 = (row11 >> 43) & 1
    alive44 = (row11 >> 44) & 1
    alive45 = (row11 >> 45) & 1
    alive46 = (row11 >> 46) & 1
    alive47 = (row11 >> 47) & 1
    alive48 = (row11 >> 48) & 1
    alive49 = (row11 >> 49) & 1
    alive50 = (row11 >> 50) & 1
    alive51 = (row11 >> 51) & 1
    alive52 = (row11 >> 52) & 1
    alive53 = (row11 >> 53) & 1
    alive54 = (row11 >> 54) & 1
    alive55 = (row11 >> 55) & 1
    alive56 = (row11 >> 56) & 1
    alive57 = (row11 >> 57) & 1
    alive58 = (row11 >> 58) & 1
    alive59 = (row11 >> 59) & 1
    alive60 = (row11 >> 60) & 1
    alive61 = (row11 >> 61) & 1
    alive62 = (row11 >> 62) & 1
    alive63 = (row11 >> 63) & 1

    # Finally, do the gol logic
    # Note our sums include the central cell
    # So it's alive ? sum == 3 || sum == 4 : sum == 3
    # i.e. (alive & (sum == 4)) | sum == 3

    alive0 = (alive0 & (sum_1 == 4)) | (sum_1 == 3)
    alive1 = (alive1 & (sum0 == 4)) | (sum0 == 3)
    alive2 = (alive2 & (sum1 == 4)) | (sum1 == 3)
    alive3 = (alive3 & (sum2 == 4)) | (sum2 == 3)
    alive4 = (alive4 & (sum3 == 4)) | (sum3 == 3)
    alive5 = (alive5 & (sum4 == 4)) | (sum4 == 3)
    alive6 = (alive6 & (sum5 == 4)) | (sum5 == 3)
    alive7 = (alive7 & (sum6 == 4)) | (sum6 == 3)
    alive8 = (alive8 & (sum7 == 4)) | (sum7 == 3)
    alive9 = (alive9 & (sum8 == 4)) | (sum8 == 3)
    alive10 = (alive10 & (sum9 == 4)) | (sum9 == 3)
    alive11 = (alive11 & (sum10 == 4)) | (sum10 == 3)
    alive12 = (alive12 & (sum11 == 4)) | (sum11 == 3)
    alive13 = (alive13 & (sum12 == 4)) | (sum12 == 3)
    alive14 = (alive14 & (sum13 == 4)) | (sum13 == 3)
    alive15 = (alive15 & (sum14 == 4)) | (sum14 == 3)
    alive16 = (alive16 & (sum15 == 4)) | (sum15 == 3)
    alive17 = (alive17 & (sum16 == 4)) | (sum16 == 3)
    alive18 = (alive18 & (sum17 == 4)) | (sum17 == 3)
    alive19 = (alive19 & (sum18 == 4)) | (sum18 == 3)
    alive20 = (alive20 & (sum19 == 4)) | (sum19 == 3)
    alive21 = (alive21 & (sum20 == 4)) | (sum20 == 3)
    alive22 = (alive22 & (sum21 == 4)) | (sum21 == 3)
    alive23 = (alive23 & (sum22 == 4)) | (sum22 == 3)
    alive24 = (alive24 & (sum23 == 4)) | (sum23 == 3)
    alive25 = (alive25 & (sum24 == 4)) | (sum24 == 3)
    alive26 = (alive26 & (sum25 == 4)) | (sum25 == 3)
    alive27 = (alive27 & (sum26 == 4)) | (sum26 == 3)
    alive28 = (alive28 & (sum27 == 4)) | (sum27 == 3)
    alive29 = (alive29 & (sum28 == 4)) | (sum28 == 3)
    alive30 = (alive30 & (sum29 == 4)) | (sum29 == 3)
    alive31 = (alive31 & (sum30 == 4)) | (sum30 == 3)
    alive32 = (alive32 & (sum31 == 4)) | (sum31 == 3)
    alive33 = (alive33 & (sum32 == 4)) | (sum32 == 3)
    alive34 = (alive34 & (sum33 == 4)) | (sum33 == 3)
    alive35 = (alive35 & (sum34 == 4)) | (sum34 == 3)
    alive36 = (alive36 & (sum35 == 4)) | (sum35 == 3)
    alive37 = (alive37 & (sum36 == 4)) | (sum36 == 3)
    alive38 = (alive38 & (sum37 == 4)) | (sum37 == 3)
    alive39 = (alive39 & (sum38 == 4)) | (sum38 == 3)
    alive40 = (alive40 & (sum39 == 4)) | (sum39 == 3)
    alive41 = (alive41 & (sum40 == 4)) | (sum40 == 3)
    alive42 = (alive42 & (sum41 == 4)) | (sum41 == 3)
    alive43 = (alive43 & (sum42 == 4)) | (sum42 == 3)
    alive44 = (alive44 & (sum43 == 4)) | (sum43 == 3)
    alive45 = (alive45 & (sum44 == 4)) | (sum44 == 3)
    alive46 = (alive46 & (sum45 == 4)) | (sum45 == 3)
    alive47 = (alive47 & (sum46 == 4)) | (sum46 == 3)
    alive48 = (alive48 & (sum47 == 4)) | (sum47 == 3)
    alive49 = (alive49 & (sum48 == 4)) | (sum48 == 3)
    alive50 = (alive50 & (sum49 == 4)) | (sum49 == 3)
    alive51 = (alive51 & (sum50 == 4)) | (sum50 == 3)
    alive52 = (alive52 & (sum51 == 4)) | (sum51 == 3)
    alive53 = (alive53 & (sum52 == 4)) | (sum52 == 3)
    alive54 = (alive54 & (sum53 == 4)) | (sum53 == 3)
    alive55 = (alive55 & (sum54 == 4)) | (sum54 == 3)
    alive56 = (alive56 & (sum55 == 4)) | (sum55 == 3)
    alive57 = (alive57 & (sum56 == 4)) | (sum56 == 3)
    alive58 = (alive58 & (sum57 == 4)) | (sum57 == 3)
    alive59 = (alive59 & (sum58 == 4)) | (sum58 == 3)
    alive60 = (alive60 & (sum59 == 4)) | (sum59 == 3)
    alive61 = (alive61 & (sum60 == 4)) | (sum60 == 3)
    alive62 = (alive62 & (sum61 == 4)) | (sum61 == 3)
    alive63 = (alive63 & (sum62 == 4)) | (sum62 == 3)

    # Pack bits
    result = (
        (alive0 << 0) | (alive1 << 1) | (alive2 << 2) | (alive3 << 3) | (alive4 << 4) | (alive5 << 5) | (alive6 << 6) | (alive7 << 7) |
        (alive8 << 8) | (alive9 << 9) | (alive10 << 10) | (alive11 << 11) | (alive12 << 12) | (alive13 << 13) | (alive14 << 14) | (alive15 << 15) | (alive16 << 16) |
        (alive17 << 17) | (alive18 << 18) | (alive19 << 19) | (alive20 << 20) | (alive21 << 21) | (alive22 << 22) | (alive23 << 23) | (alive24 << 24) |
        (alive25 << 25) | (alive26 << 26) | (alive27 << 27) | (alive28 << 28) | (alive29 << 29) | (alive30 << 30) | (alive31 << 31) |
        (alive32 << 32) | (alive33 << 33) | (alive34 << 34) | (alive35 << 35) | (alive36 << 36) | (alive37 << 37) | (alive38 << 38) | (alive39 << 39) |
        (alive40 << 40) | (alive41 << 41) | (alive42 << 42) | (alive43 << 43) | (alive44 << 44) | (alive45 << 45) | (alive46 << 46) | (alive47 << 47) |
        (alive48 << 48) | (alive49 << 49) | (alive50 << 50) | (alive51 << 51) | (alive52 << 52) | (alive53 << 53) | (alive54 << 54) | (alive55 << 55) |
        (alive56 << 56) | (alive57 << 57) | (alive58 << 58) | (alive59 << 59) | (alive60 << 60) | (alive61 << 61) | (alive62 << 62) | (alive63 << 63)
    )

    out_offsets = col_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_mask = (out_offsets >= 0) & (out_offsets < ((N + 63) // 64))

    tl.store(out_ptr + row_id * row_stride + out_offsets, result, mask=out_mask)

def gol_triton_64bit_1d(x: torch.Tensor, BLOCK_SIZE: int = None):
    BLOCK_SIZE = BLOCK_SIZE or 512

    output = torch.empty_like(x)

    def grid(meta):
        bs = meta['BLOCK_SIZE']
        return (x.shape[0], triton.cdiv(x.shape[1], bs))
    
    gol_triton_64bit_1d_kernel[grid](x, output, x.stride(0), x.shape[0], BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    return output

def longlong_encode(x: torch.Tensor):
    assert x.shape[1] % 64 == 0
    assert x.is_contiguous()
    x = bit_encode(x)
    # out = torch.empty(0, dtype=torch.uint32, device=x.device)
    # out.set_(x.untyped_storage(), x, (x.shape[0] , x.shape[1] // 4))
    return x.view(torch.uint64)

def longlong_decode(x: torch.Tensor):
    assert x.is_contiguous()
    # out = torch.empty(0, dtype=torch.uint8, device=x.device)
    # out.set_(x.untyped_storage(), x, (x.shape[0] , x.shape[1] *4))
    out = x.view(torch.uint8)
    return bit_decode(out)

# %%


def visualize_heatmap(x: torch.Tensor, title: str = "Heatmap"):
    if x.dim() != 2:
        raise ValueError(f"Input tensor must be 2D, but got shape {x.shape}")
    
    # Convert tensor to numpy array for plotting. Ensure it's on CPU.
    x_np = x.cpu().numpy()

    plt.figure(figsize=(6, 6))
    
    # Use 'Greys' colormap: 0 (white) to 1 (black).
    # 'origin='upper'' ensures (0,0) is at the top-left, standard for image data.
    # vmin/vmax ensure the colormap is consistently scaled for 0 and 1 values.
    plt.imshow(x_np, cmap='Greys', origin='upper', vmin=0, vmax=1)
    
    # Add a colorbar for clarity, specifically showing 0 and 1 values.
    cbar = plt.colorbar(ticks=[0, 1])
    cbar.set_label("Value (0: Dead, 1: Live)")

    plt.title(title)
    plt.xlabel("Width")
    plt.ylabel("Height")
    
    # Set ticks to align with grid cells for better readability.
    plt.xticks(range(x_np.shape[1]))
    plt.yticks(range(x_np.shape[0]))
    
    # Add grid lines to clearly delineate cells.
    plt.grid(True, which='both', color='lightgrey', linestyle='-', linewidth=0.5)
    
    # Adjust plot to ensure everything fits without overlapping.
    plt.tight_layout()
    plt.show()


# %%

x = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]).to(torch.int8).to(device)
x = torch.zeros((64, 64)).to(torch.int8).to(device)
x[2, 1] = 1
x[2, 2] = 1
x[2, 3] = 1
# visualize_heatmap(x[:6, : 6])

# x = gol_torch_conv2d_compiled(x)
# visualize_heatmap(x[:6, : 6])

# x = gol_triton_1d(x)
# visualize_heatmap(x[:6, : 6])

# x = gol_triton_2d(x)
# visualize_heatmap(x[:6, : 6])

# x = gol_cuda(x)
# visualize_heatmap(x[:6, : 6])

x = gol_cuda_shared_memory(x)
visualize_heatmap(x[:6, : 6])


# x = longlong_encode(x)
# x = gol_triton_64bit_1d(x)
# x = longlong_decode(x)
# visualize_heatmap(x[:6, : 6])


# %%

def get_roofline_ms(N):
    return 2 * N * N / 696000000000 * 1000

def get_bit_roofline_ms(N):
    return get_roofline_ms(N) / 8

# %%
# Key results

IS_ROOFLINE = False

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(2, 32 + 1, 2)],
        line_arg='provider',
        line_vals=['torch', 'compiled_torch', 'triton', 'cuda', 'triton_32bit'],
        line_names=['Torch', 'Compiled Torch', 'Triton', 'Cuda', 'Triton 32bit'],
        ylabel='ms',
        plot_name='gol main',
        args={}
    ))
def benchmark(provider, N):
    print(provider, N)
    # create data
    x_shape = (N, N)
    x = (torch.rand(x_shape, device=device) < 0.5).to(torch.int8).to(device)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gol_torch_conv2d(x), quantiles=quantiles, rep=500)
    elif provider == 'compiled_torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gol_torch_conv2d_compiled(x), quantiles=quantiles, rep=500)
    elif provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gol_triton_1d(x), quantiles=quantiles, rep=500)
    elif provider == 'cuda':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gol_cuda(x), quantiles=quantiles, rep=500)
    elif provider == 'cuda2':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gol_cuda_shared_memory(x), quantiles=quantiles, rep=500)
    elif provider == 'triton_8bit':
        x = bit_encode(x)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gol_triton_8bit_1d(x), quantiles=quantiles, rep=500)
    elif provider == 'triton_32bit':
        x = long_encode(x)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gol_triton_32bit_1d(x), quantiles=quantiles, rep=500)
    elif provider == 'triton_64bit':
        x = longlong_encode(x)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gol_triton_64bit_1d(x), quantiles=quantiles, rep=500)

    if IS_ROOFLINE:
        roofline_ms = get_roofline_ms(N) if 'bit' not in provider else get_bit_roofline_ms(N)
        return roofline_ms / ms
    else:
        return ms, min_ms, max_ms

benchmark.run(print_data=True, show_plots=True)

# %%
# triton vs cuda comparisons
# It's important to remember that triton BLOCK_SIZE is
# not the same as cuda thread block size.
# Essentially triton always uses a thread block size of (32*num_warps, 1, 1)
# If BLOCK_SIZE is larger than that, it starts generates kernel's
# that process more than one element.
# Thus a like-for-like comparison requires setting num_warps = BLOCK_SIZE / 32
# rather than the default of 4.

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[2**16],
        line_arg='provider',
        line_vals=['triton', 'cuda'],
        line_names=['Triton', 'Cuda'],
        ylabel='ms',
        plot_name='gol main',
        args={}
    ))
def benchmark(provider, N):
    print(provider, N)
    # create data
    x_shape = (N, N)
    x = torch.randint(0, 1, x_shape, device=device, dtype=torch.int8)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gol_triton_1d(x, BLOCK_SIZE=1024, num_warps = 32), quantiles=quantiles, rep=500)
    elif provider == 'cuda':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gol_cuda(x, BLOCK_SIZE_COL=256, BLOCK_SIZE_ROW=1), quantiles=quantiles, rep=500)

    return ms, min_ms, max_ms

benchmark.run(print_data=True, show_plots=True)

# %% 
# bit variants
# This benchmark function is modified to ramp up to larger N than the non bit-compressed takes

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[8*512 * i for i in range(2, 32, 2)],
        line_arg='provider',
        line_vals=['triton_8bit', 'triton_32bit', 'triton_64bit'],
        line_names=['Triton 8bit', 'Triton 32bit', 'Triton 64bit'],
        ylabel='ms',
        plot_name='gol bit variants',
        args={}
    ))
def benchmark(provider, N):
    # create data
    x = torch.randint(0, 255, (N, N // 8), device=device, dtype=torch.uint8)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'triton_8bit':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gol_triton_8bit_1d(x), quantiles=quantiles, rep=500)
    elif provider == 'triton_32bit':
        x = x.view(torch.uint32)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gol_triton_32bit_1d(x), quantiles=quantiles, rep=500)
    elif provider == 'triton_64bit':
        x = x.view(torch.uint64)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gol_triton_64bit_1d(x), quantiles=quantiles, rep=500)
    else:
        raise ValueError(f"Invalid provider: {provider}")
    return ms, min_ms, max_ms

benchmark.run(print_data=True, show_plots=True)

# %%
# Torch variants

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(2, 16, 2)],
        line_arg='provider',
        line_vals=['torch', 'compiled_torch', 'triton', 'torch_f16', 'compiled_torch_f16', 'torch_sum', 'compiled_torch_sum'],
        line_names=['Torch', 'Compiled Torch', 'Triton', 'Torch F16', 'Compiled Torch F16', 'Torch Sum', 'Compiled Torch Sum'],
        ylabel='ms',
        plot_name='gol torch variants',
        args={}
    ))
def benchmark(provider, N):
    # create data
    x_shape = (N, N)
    x = (torch.rand(x_shape, device=device) < 0.5).to(torch.int8).to(device)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gol_torch_conv2d(x), quantiles=quantiles, rep=500)
    elif provider == 'compiled_torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gol_torch_conv2d_compiled(x), quantiles=quantiles, rep=500)
    elif provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gol_triton_1d(x), quantiles=quantiles, rep=500)
    elif provider == 'torch_f16':
        x = x.to(torch.float16)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gol_torch_conv2d_f16(x), quantiles=quantiles, rep=500)
    elif provider == 'compiled_torch_f16':
        x = x.to(torch.float16)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gol_torch_conv2d_f16_compiled(x), quantiles=quantiles, rep=500)
    elif provider == 'torch_sum':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gol_torch_sum(x), quantiles=quantiles, rep=500)
    elif provider == 'compiled_torch_sum':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gol_torch_sum_compiled(x), quantiles=quantiles, rep=500)
    else:
        raise ValueError(f"Invalid provider: {provider}")
    return ms, min_ms, max_ms

benchmark.run(print_data=True, show_plots=True)


# %% 
# cuda variants

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(2, 32 + 1, 2)],
        line_arg='provider',
        line_vals=['compiled_torch', 'cuda', 'cuda_shared_memory'],
        line_names=['Compiled Torch', 'Cuda', 'Cuda Shared Memory'],
        ylabel='ms',
        plot_name='gol bit variants',
        args={}
    ))
def benchmark(provider, N):
    print(provider, N)
    # create data
    x_shape = (N, N)
    x = (torch.rand(x_shape, device=device) < 0.5).to(torch.int8).to(device)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'compiled_torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gol_torch_conv2d_compiled(x), quantiles=quantiles, rep=500)
    elif provider == 'cuda':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gol_cuda(x), quantiles=quantiles, rep=500)
    elif provider == 'cuda_shared_memory':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gol_cuda_shared_memory(x), quantiles=quantiles, rep=500)
    else:
        raise ValueError(f"Invalid provider: {provider}")
    return ms, min_ms, max_ms

benchmark.run(print_data=True, show_plots=True)

# %%

# Test 1d block sizes

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(2, 16, 2)],
        line_arg='provider',
        line_vals=[256 * (2**i) for i in range(0, 6)],
        line_names=[str(256 * (2**i)) for i in range(0, 6)],
        # styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
        ylabel='ms',
        plot_name='gol 1d block sizes',
        args={}
    ))
def benchmark(provider, N):
    # create data
    x_shape = (N, N)
    x = (torch.rand(x_shape, device=device) < 0.5).to(torch.int8).to(device)
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: gol_triton_1d(x, BLOCK_SIZE=provider), quantiles=quantiles, rep=500)
    return ms, min_ms, max_ms

benchmark.run(print_data=True, show_plots=True)

# %%
# Test 2d block sizes

block_sizes = [
    (1 * (2**i), 1 * (2**j)) for i in range(0, 9) for j in range(0 , 12)
    if i + j < 11
]
def block_str(size):
    return str(size[0]) + "x" + str(size[1])
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(2, 16, 2)],
        line_arg='block_sizes',
        line_vals=block_sizes,
        line_names=[block_str(size) for size in block_sizes],
        # styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
        ylabel='ms',
        plot_name='gol 2d block sizes',
        args={}
    ))
def benchmark(block_sizes, N):
    # create data
    x_shape = (N, N)
    x = (torch.rand(x_shape, device=device) < 0.5).to(torch.int8).to(device)
    quantiles = [0.5, 0.2, 0.8]
    # ms, min_ms, max_ms = triton.testing.do_bench(lambda: gol_triton_2d(x, BLOCK_SIZE_ROW=block_sizes[0], BLOCK_SIZE_COL=block_sizes[1]), quantiles=quantiles, rep=500)
    print(block_sizes)
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: gol_cuda(x, BLOCK_SIZE_ROW=block_sizes[0], BLOCK_SIZE_COL=block_sizes[1]), quantiles=quantiles, rep=500)
    return ms, min_ms, max_ms

result = benchmark.run(return_df=True)

result

# %%

row_sizes = set([size[0] for size in block_sizes])
col_sizes = set([size[1] for size in block_sizes])

data={}

for row_size in row_sizes:
    for col_size in col_sizes:
        s = block_str((row_size, col_size))
        if s in result:
            data[(row_size, col_size)] = result[s][6].item()

import numpy as np

# Sort the unique row and column sizes for consistent plotting order
sorted_row_sizes = sorted(list(row_sizes))
sorted_col_sizes = sorted(list(col_sizes))

# Create a 2D array to store the heatmap data
heatmap_matrix = np.zeros((len(sorted_row_sizes), len(sorted_col_sizes)))

# Populate the heatmap matrix with median execution times
for i, r_size in enumerate(sorted_row_sizes):
    for j, c_size in enumerate(sorted_col_sizes):
        # The 'data' dictionary is already populated with the median 'ms' values
        heatmap_matrix[i, j] = data.get((r_size, c_size), float('nan'))

# Plot the heatmap
plt.figure(figsize=(8, 6))
# Use 'viridis' colormap, 'lower' origin to have (0,0) at bottom-left, and 'auto' aspect ratio
plt.imshow(heatmap_matrix, cmap='viridis', origin='lower', aspect='auto')

# Set ticks and labels for the axes
plt.xticks(np.arange(len(sorted_col_sizes)), sorted_col_sizes)
plt.yticks(np.arange(len(sorted_row_sizes)), sorted_row_sizes)

plt.xlabel("BLOCK_SIZE_COL")
plt.ylabel("BLOCK_SIZE_ROW")
plt.title("Median Execution Time (ms) for gol_cuda Block Sizes")

# Add a colorbar to indicate the scale of execution times
cbar = plt.colorbar()
cbar.set_label("Median Time (ms)")

# Add text annotations for the values in each cell
for i in range(len(sorted_row_sizes)):
    for j in range(len(sorted_col_sizes)):
        plt.text(j, i, f"{heatmap_matrix[i, j]:.2f}",
                 ha="center", va="center", color="w", fontsize=9) # White text for better contrast on viridis

plt.tight_layout() # Adjust plot to ensure everything fits without overlapping
plt.show()