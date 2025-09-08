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
    y = torch.nn.functional.conv2d(x[None, :, :].to(dtype=torch.float16), WEIGHT_F16, padding=1)[0].to(torch.int8)
    return torch.where(x > 0, (y == 2) | (y == 3), (y == 3)).to(torch.int8)

@torch.compile
def gol_torch_conv2d_compiled(x: torch.Tensor):
    return gol_torch_conv2d(x)

# %%
def gol_torch_conv2d_f16(x: torch.Tensor):
    y = torch.nn.functional.conv2d(x[None, :, :].to(dtype=torch.float16), WEIGHT_F16, padding=1)[0]
    return torch.where(x > 0, (y == 2) | (y == 3), (y == 3)).to(torch.float16)

@torch.compile
def gol_torch_conv2d_f16_compiled(x: torch.Tensor):
    return gol_torch_conv2d_f16(x)

# %%
# Sum neighbors via slicing
def gol_torch_sum(x: torch.Tensor):
    y = torch.zeros_like(x)
    p = [
        (slice(1, None), slice(None, -1)),
        (slice(None, None), slice(None, None)),
        (slice(None, -1), slice(1, None)),
    ]
    for a, b in p:
        for c, d in p:
            y[a, c] += x[b, d]

    return torch.where(x == 1, (y == 2) | (y == 3), (y == 3)).to(torch.int8)    

@torch.compile
def gol_torch_sum_compiled(x: torch.Tensor):
    return gol_torch_sum(x)

# %%

# Doesn't work: Triton doesn't support slicing yet

@triton.jit
def gol_triton_1d_slice_kernel(x_ptr, out_ptr, row_stride, N: tl.int64, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    col_id = tl.program_id(1)

    offsets = col_id * (BLOCK_SIZE - 2) - 1 + tl.arange(0, BLOCK_SIZE)

    indexes = x_ptr + row_id * row_stride + offsets
    mask = (offsets < N) & (offsets >= 0)


    row0 = tl.load(indexes - row_stride, mask=mask, other=0) if row_id > 0 else tl.zeros((BLOCK_SIZE,), dtype=tl.int8)
    row1 = tl.load(indexes, mask=mask, other=0)
    row2 = tl.load(indexes + row_stride, mask=mask, other=0) if row_id < N - 1 else tl.zeros((BLOCK_SIZE,), dtype=tl.int8)

    sum = row0 + row1 + row2
    # Oops: Triton doesn't support slicing yet
    sum2 = sum[:-2] + sum[1:-1] + sum[2:]
    current = row1[1:-1]

    result = tl.where(current > 0, (sum2 == 2) | (sum2 == 3), sum2 == 3).to(tl.int8)

    tl.store(out_ptr + row_id * row_stride + offsets, result, mask=mask, other=0)

def gol_triton_1d_slice(x: torch.Tensor):
    BLOCK_SIZE = 256

    output = torch.empty_like(x)

    def grid(meta):
        bs = meta['BLOCK_SIZE']
        return (x.shape[0], triton.cdiv(x.shape[1], bs - 2))
    
    gol_triton_1d_slice_kernel[grid](x, output, x.stride(0), x.shape[0], BLOCK_SIZE=BLOCK_SIZE)

    return output

# %%

@triton.jit
def gol_triton_1d_kernel(x_ptr, out_ptr, row_stride, N: tl.int64, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    col_id = tl.program_id(1)



    offsets0 = col_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) - 1
    offsets1 = col_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets2 = col_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) + 1

    mask0 = (offsets0 >= 0) & (offsets0 < N)
    mask1 = (offsets1 >= 0) & (offsets1 < N)
    mask2 = (offsets2 >= 0) & (offsets2 < N)

    row_ptr = x_ptr + row_id * row_stride

    if row_id > 0:
        row00 = tl.load(row_ptr + offsets0 - row_stride, mask=mask0, other=0)
        row01 = tl.load(row_ptr + offsets1 - row_stride, mask=mask1, other=0)
        row02 = tl.load(row_ptr + offsets2 - row_stride, mask=mask2, other=0)
    else:
        row00 = tl.zeros((BLOCK_SIZE,), dtype=tl.int8)
        row01 = tl.zeros((BLOCK_SIZE,), dtype=tl.int8)
        row02 = tl.zeros((BLOCK_SIZE,), dtype=tl.int8)
    row10 = tl.load(row_ptr + offsets0, mask=mask0, other=0)
    row11 = tl.load(row_ptr + offsets1, mask=mask1, other=0)
    row12 = tl.load(row_ptr + offsets2, mask=mask2, other=0)
    if row_id < N - 1:
        row20 = tl.load(row_ptr + offsets0 + row_stride, mask=mask0, other=0)
        row21 = tl.load(row_ptr + offsets1 + row_stride, mask=mask1, other=0)
        row22 = tl.load(row_ptr + offsets2 + row_stride, mask=mask2, other=0)
    else:
        row20 = tl.zeros((BLOCK_SIZE,), dtype=tl.int8)
        row21 = tl.zeros((BLOCK_SIZE,), dtype=tl.int8)
        row22 = tl.zeros((BLOCK_SIZE,), dtype=tl.int8)

    sum = row00 + row01 + row02 + row10 +  row12 + row20 + row21 + row22

    result = tl.where(row11 > 0, (sum == 2) | (sum == 3), sum == 3).to(tl.int8)

    out_offsets = col_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_mask = (out_offsets >= 0) & (out_offsets < N)

    tl.store(out_ptr + row_id * row_stride + out_offsets, result, mask=out_mask)

def gol_triton_1d(x: torch.Tensor, BLOCK_SIZE: int = None):
    # I don't understand block size tuning yet.
    # There seems to be a significant performance difference between 4096 and 8192.
    BLOCK_SIZE = BLOCK_SIZE or 1024

    output = torch.empty_like(x)

    def grid(meta):
        bs = meta['BLOCK_SIZE']
        return (x.shape[0], triton.cdiv(x.shape[1], bs))
    
    gol_triton_1d_kernel[grid](x, output, x.stride(0), x.shape[0], BLOCK_SIZE=BLOCK_SIZE)

    return output

# %%

@triton.jit
def gol_triton_2d_kernel(x_ptr, out_ptr, row_stride, N: tl.int64, BLOCK_SIZE_ROW: tl.constexpr, BLOCK_SIZE_COL: tl.constexpr):
    row_id = tl.program_id(0)
    col_id = tl.program_id(1)


    col_offsets0 = (col_id * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL) - 1)[None, :]
    col_offsets1 = (col_id * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL))[None, :]
    col_offsets2 = (col_id * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL) + 1)[None, :]

    row_offsets0 = (row_id * BLOCK_SIZE_ROW + tl.arange(0, BLOCK_SIZE_ROW) - 1)[:, None]
    row_offsets1 = (row_id * BLOCK_SIZE_ROW + tl.arange(0, BLOCK_SIZE_ROW))[:, None]
    row_offsets2 = (row_id * BLOCK_SIZE_ROW + tl.arange(0, BLOCK_SIZE_ROW) + 1)[:, None]

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
        triton.cdiv(x.shape[0], BLOCK_SIZE_ROW),
        triton.cdiv(x.shape[1], BLOCK_SIZE_COL),
    )

    gol_triton_2d_kernel[grid](x, output, x.stride(0), x.shape[0], BLOCK_SIZE_ROW=BLOCK_SIZE_ROW, BLOCK_SIZE_COL=BLOCK_SIZE_COL)

    return output

# %%
import torch
from torch.utils.cpp_extension import load_inline

ext = load_inline(
    name="gol_ext",
    cpp_sources="",            # no separate C++ binding file
    cuda_sources=[open("kernel.cpp").read()],   # contains both kernel and PYBIND11 module
    with_cuda=True,
    extra_cuda_cflags=["-O3"],
    verbose=True,
)

def gol_cuda(x: torch.Tensor, BLOCK_SIZE_ROW: int = None, BLOCK_SIZE_COL: int = None):
    if BLOCK_SIZE_ROW is None and BLOCK_SIZE_COL is None:
        BLOCK_SIZE_ROW = 4
        BLOCK_SIZE_COL = 64
    output = torch.empty_like(x)
    ext.gol(x, output, BLOCK_SIZE_ROW, BLOCK_SIZE_COL)
    return output
# %%
import torch
from torch.utils.cpp_extension import load_inline


ext2 = load_inline(
    name="gol2_ext",
    cpp_sources="",            # no separate C++ binding file
    cuda_sources=[open("kernel2.cpp").read()],   # contains both kernel and PYBIND11 module
    with_cuda=True,
    extra_cuda_cflags=["-O3"],
    verbose=True,
)

def gol_cuda_shared_memory(x: torch.Tensor):
    output = torch.empty_like(x)
    ext2.gol(x, output)
    return output

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
visualize_heatmap(x)
x = gol_cuda_shared_memory(x)
visualize_heatmap(x)

# %%
# Key results

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(2, 16, 2)],
        line_arg='provider',
        line_vals=['torch', 'compiled_torch', 'triton', 'cuda', 'cuda2'],
        line_names=['Torch', 'Compiled Torch', 'Triton', 'CUDA', 'CUDA (shared memory)'],
        ylabel='ms',
        plot_name='gol',
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
    elif provider == 'cuda':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gol_cuda(x), quantiles=quantiles, rep=500)
    elif provider == 'cuda2':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gol_cuda_shared_memory(x), quantiles=quantiles, rep=500)
    else:
        raise ValueError(f"Invalid provider: {provider}")
    return ms, min_ms, max_ms

benchmark.run(print_data=True, show_plots=True)

# %%
# Test variants

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(2, 16, 2)],
        line_arg='provider',
        line_vals=['torch', 'compiled_torch', 'triton', 'torch_f16', 'compiled_torch_f16', 'torch_sum', 'compiled_torch_sum'],
        line_names=['Torch', 'Compiled Torch', 'Triton', 'Torch F16', 'Compiled Torch F16', 'Torch Sum', 'Compiled Torch Sum'],
        ylabel='ms',
        plot_name='gol',
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
        plot_name='gol',
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
        plot_name='gol',
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



# %%

def main():
    print("Hello from gol!")


if __name__ == "__main__":
    main()

