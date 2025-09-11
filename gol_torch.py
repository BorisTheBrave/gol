import torch


# %%
# Use torch's built in convolution
# We have to convert to float16 because the convolution doesn't support int8

WEIGHT_F16 = torch.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]])[None, None, :, :].to(torch.float16)

def gol_torch_conv2d(x: torch.Tensor) -> torch.Tensor:
    w = WEIGHT_F16.to(x.device)
    y = torch.nn.functional.conv2d(x[None, :, :].to(dtype=torch.float16), )[0].to(torch.int8)
    y = torch.nn.functional.pad(y, (1, 1, 1, 1), value=0)
    y = torch.where(x > 0, (y == 2) | (y == 3), (y == 3)).to(torch.int8)
    return y


@torch.compile
def gol_torch_conv2d_compiled(x: torch.Tensor) -> torch.Tensor:
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


def gol_torch_sum(x: torch.Tensor) -> torch.Tensor:
    y = x[2:] + x[1:-1] + x[:-2]
    z = y[:, 2:] + y[:, 1:-1] + y[:, :-2]
    z = torch.nn.functional.pad(z, (1, 1, 1, 1), value=0)
    return ((x == 1) & (z == 4)) | (z == 3).to(torch.int8)


@torch.compile
def gol_torch_sum_compiled(x: torch.Tensor):
    return gol_torch_sum(x)
