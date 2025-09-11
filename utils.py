import torch
import triton
import matplotlib.pyplot as plt


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


# %%
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
