# %%
# %load_ext autoreload
# %autoreload 2

# %%
import torch
import triton
import matplotlib.pyplot as plt
import numpy as np

from utils import visualize_heatmap, bit_encode, bit_decode, long_encode, long_decode, longlong_encode, longlong_decode
from gol_torch import gol_torch_conv2d, gol_torch_conv2d_compiled, gol_torch_conv2d_f16, gol_torch_conv2d_f16_compiled, gol_torch_sum, gol_torch_sum_compiled
from gol_cuda import gol_cuda, gol_cuda_shared_memory, gol_cuda_wideload, gol_cuda_grouped, gol_cuda_bitpacked, gol_cuda_bitpacked_64, gol_cuda_grouped_bitpacked_64, gol_cuda_grouped_bitpacked_64_multistep
from gol_triton import gol_triton_1d, gol_triton_2d, gol_triton_8bit_1d, gol_triton_32bit_1d, gol_triton_64bit_1d, gol_triton_2d_kernel

device = torch.device('cuda:0')

# %%
# Test data
x = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]).to(torch.int8).to(device)
x = torch.zeros((1024, 1024)).to(torch.int8).to(device)
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

# x = gol_cuda_shared_memory(x)
# visualize_heatmap(x[:6, : 6])

# x = gol_cuda_wideload(x)
# visualize_heatmap(x[:6, : 6])


# x = longlong_encode(x)
# x = gol_cuda_grouped_bitpacked_64(x)
# x = longlong_decode(x)
# visualize_heatmap(x[:6, : 6])


x = longlong_encode(x)
x = gol_cuda_grouped_bitpacked_64_multistep(x, BLOCK_SIZE_ROW=16, BLOCK_SIZE_COL=1024)
x = longlong_decode(x)
visualize_heatmap(x[:6, : 6])


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
        line_vals=['torch', 'compiled_torch', 'triton', 'cuda', 'triton_32bit', 'cuda_shared_memory', 'cuda_wideload'],
        line_names=['Torch', 'Compiled Torch', 'Triton', 'Cuda', 'Triton 32bit', 'Cuda Shared Memory', 'Cuda Wideload'],
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
    elif provider == 'cuda_shared_memory':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gol_cuda_shared_memory(x), quantiles=quantiles, rep=500)
    elif provider == 'cuda_wideload':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gol_cuda_wideload(x), quantiles=quantiles, rep=500)
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
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gol_triton_1d(x, BLOCK_SIZE=256, num_warps=8, num_stages=1), quantiles=quantiles, rep=500)
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
        # x_vals=[8*512 * i for i in range(2, 32, 2)],
        x_vals=[2**16],
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
        x_vals=[2**16],
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
        # x_vals=[512 * i for i in range(2, 32 + 1, 2)],
        x_vals=[2**16],
        line_arg='provider',
        line_vals=['compiled_torch', 'cuda', 'cuda_shared_memory', 'cuda_wideload', 'cuda_bitpacked'],
        line_names=['Compiled Torch', 'Cuda', 'Cuda Shared Memory', 'Cuda Wideload', 'Cuda Bitpacked'],
        ylabel='ms',
        plot_name='gol cuda variants',
        args={}
    ))
def benchmark(provider, N):
    print(provider, N)
    # create data
    x_shape = (N, N)
    x = torch.randint(0, 1, x_shape, device=device, dtype=torch.int8)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'compiled_torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gol_torch_conv2d_compiled(x), quantiles=quantiles, rep=500)
    elif provider == 'cuda':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gol_cuda(x), quantiles=quantiles, rep=500)
    elif provider == 'cuda_shared_memory':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gol_cuda_shared_memory(x), quantiles=quantiles, rep=500)
    elif provider == 'cuda_wideload':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gol_cuda_wideload(x), quantiles=quantiles, rep=500)
    elif provider == 'cuda_bitpacked':
        x = bit_encode(x)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gol_cuda_bitpacked(x), quantiles=quantiles, rep=500)
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

import math

TEST_FN = gol_cuda_grouped_bitpacked_64_multistep
STEPS = 32
# CUDA limit on threads is 1024 (10 bits)
# grouped implementations add more bits for thie group
# and bitpacked implementations add 3 or 6 bits
# There seems to be a limit on multistep for allocating shared memory too.
MAX_BLOCK_SIZE = {
    gol_cuda: 10, 
    gol_cuda_wideload: 12,
    gol_triton_2d: 12,
    gol_cuda_grouped: 12,
    gol_cuda_bitpacked: 10,
    gol_cuda_bitpacked_64: 10,
    gol_cuda_grouped_bitpacked_64: 10 + 2 + 6,
    gol_cuda_grouped_bitpacked_64_multistep: 10 + 2 + 6 - 1,
}[TEST_FN]
MIN_BLOCK_SIZE_ROW = {
    gol_cuda_grouped_bitpacked_64: 2,
    gol_cuda_grouped_bitpacked_64_multistep: max(4, int(math.log2(STEPS))+ 2),
}.get(TEST_FN, 0)
MIN_BLOCK_SIZE_COL = {
    gol_cuda_wideload: 4,
    gol_cuda_grouped: 4,
    gol_cuda_grouped_bitpacked_64: 6,
    gol_cuda_grouped_bitpacked_64_multistep: 8,
}.get(TEST_FN, 0)
IS_TRITON = TEST_FN == gol_triton_2d

block_sizes = [
    (2**i, 2**j) for i in range(0, MAX_BLOCK_SIZE) for j in range(0, MAX_BLOCK_SIZE + 1)
    # CUDA has a limit on the number of threads
    if i + j <= MAX_BLOCK_SIZE
    # Warp size 32, rarely any point trying lower than that
    if i + j >= 5
    # Some kernels have frequirements on the multiple of the block size.
    if j >= MIN_BLOCK_SIZE_COL
    if i >= MIN_BLOCK_SIZE_ROW
]
def block_str(size):
    return str(size[0]) + "x" + str(size[1])
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        # x_vals=[512 * i for i in range(2, 16, 2)],
        x_vals=[2**16],
        line_arg='block_sizes',
        line_vals=block_sizes,
        line_names=[block_str(size) for size in block_sizes],
        # styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
        ylabel='ms',
        plot_name='gol 2d block sizes',
        args={}
    ))
def benchmark(block_sizes, N, STEPS=STEPS):
    # create data
    x_shape = (N, N)
    x = (torch.randint(0, 1, x_shape, device=device, dtype=torch.int8))
    quantiles = [0.5, 0.2, 0.8]
    BLOCK_SIZE_ROW, BLOCK_SIZE_COL = block_sizes
    print("BLOCK_SIZE_ROW", BLOCK_SIZE_ROW)
    print("BLOCK_SIZE_COL", BLOCK_SIZE_COL)
    if 'bitpacked' in TEST_FN.__name__:
        x = bit_encode(x)
    kwargs = {}
    if 'multistep' in TEST_FN.__name__:
        kwargs['STEPS'] = STEPS
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: TEST_FN(
            x,
            BLOCK_SIZE_ROW=BLOCK_SIZE_ROW,
            BLOCK_SIZE_COL=BLOCK_SIZE_COL,
            **kwargs),
        quantiles=quantiles, rep=500)
    # Average over steps
    if 'multistep' in TEST_FN.__name__:
        ms = ms / STEPS
        min_ms = min_ms / STEPS
        max_ms = max_ms / STEPS
    return ms, min_ms, max_ms

result = benchmark.run(return_df=True, show_plots=False)

result

# %%

#result['1x1'][0]= float('nan')

row_sizes = set([size[0] for size in block_sizes])
col_sizes = set([size[1] for size in block_sizes])

data={}

for row_size in row_sizes:
    for col_size in col_sizes:
        s = block_str((row_size, col_size))
        if s in result:
            data[(row_size, col_size)] = result[s][0].item()

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

# Find the minimum value and its coordinates
min_value = np.nanmin(heatmap_matrix)
min_row_idx, min_col_idx = np.unravel_index(np.nanargmin(heatmap_matrix), heatmap_matrix.shape)

# Highlight the cell with the lowest value
from matplotlib.patches import Rectangle
current_ax = plt.gca()
rect = Rectangle((min_col_idx - 0.5, min_row_idx - 0.5), 1, 1,
                 linewidth=3, edgecolor='red', facecolor='none')
current_ax.add_patch(rect)

fn_name = TEST_FN.__name__
if 'multistep' in fn_name:
    fn_name = fn_name + f" (STEPS={STEPS})"

# Set ticks and labels for the axes
plt.xticks(np.arange(len(sorted_col_sizes)), sorted_col_sizes)
plt.yticks(np.arange(len(sorted_row_sizes)), sorted_row_sizes)

plt.xlabel("BLOCK_SIZE_COL")
plt.ylabel("BLOCK_SIZE_ROW")
plt.title(f"Execution Time for {fn_name} Block Sizes")

# Add a colorbar to indicate the scale of execution times
cbar = plt.colorbar()
cbar.set_label("Median Time (ms)")

# Add text annotations for the values in each cell
for i in range(len(sorted_row_sizes)):
    for j in range(len(sorted_col_sizes)):
        plt.text(j, i, f"{heatmap_matrix[i, j]:.2f}",
                 ha="center", va="center", color="w", fontsize=9) # White text for better contrast on viridis

plt.tight_layout() # Adjust plot to ensure everything fits without overlapping
plt.savefig(f"images/block_sizes_2d_{fn_name}.png")
plt.show()


# %%
x = torch.randint(0, 1, (2**16, 2**16), device=device, dtype=torch.int8)
out = torch.empty_like(x)
k = gol_triton_2d_kernel.warmup(x, out, x.stride(0), x.shape[0], BLOCK_SIZE_ROW=4, BLOCK_SIZE_COL=256, grid=(1, 1))
print(k.asm['llir'])
# %%

# Plot all results (as recorded on a A40 for N=2**16)
durations = {
    "Pytorch": 223,
    "torch.compile": 38.1,
    "Naive CUDA\n$(1\\times128)$": 26,
    "Naive Triton\n$(4\\times256)$": 22.5,
    "Grouped CUDA\n$(1\\times512)$": 14.7,
    "Bitpacked 8-bit Triton\n$(1\\times128)$": 14.9,
    "Bitpacked 32-bit Triton\n$(1\\times256)$": 5.21,
    "Bitpacked 64-bit CUDA\n$(1\\times1024)$": 1.84,
}

labels = list(durations.keys())[1:]
values = list(durations.values())[1:]
plt.text(0.5, 0.95, "(lower is better)", ha='center', va='top', transform=plt.gca().transAxes, fontsize=10)

plt.bar(labels, values, color='skyblue')
plt.ylabel("Execution Time (ms)")
plt.xlabel("Optimal block size shown as (rows$\\times$cols)")
plt.title("Game of Life Implementations Performance ($N=2^{16}$, device=A40)")
plt.xticks(rotation=45, ha='right') # Rotate labels for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7) # Add a grid for easier comparison
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()
