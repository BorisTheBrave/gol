import torch
from torch.utils.cpp_extension import load_inline


# %%
ext = None
def init_ext():
    global ext
    ext = load_inline(
        name="gol_ext",
        cpp_sources="",            # no separate C++ binding file
        cuda_sources=[open("gol_cuda.cpp").read()],   # contains both kernel and PYBIND11 module
        with_cuda=True,
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )

def gol_cuda(x: torch.Tensor, BLOCK_SIZE_ROW: int = None, BLOCK_SIZE_COL: int = None):
    if ext is None: init_ext()
    if BLOCK_SIZE_ROW is None and BLOCK_SIZE_COL is None:
        BLOCK_SIZE_ROW = 1
        BLOCK_SIZE_COL = 128
    output = torch.empty_like(x)
    ext.gol(x, output, BLOCK_SIZE_ROW, BLOCK_SIZE_COL)
    return output


# %%
ext2 = None

def init_ext2():
    global ext2
    ext2 = load_inline(
        name="gol2_ext",
        cpp_sources="",            # no separate C++ binding file
        cuda_sources=[open("gol_cuda_shared_memory.cpp").read()],   # contains both kernel and PYBIND11 module
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
ext3 = None

def init_ext3():
    global ext3
    ext3 = load_inline(
        name="gol3_ext",
        cpp_sources="",            # no separate C++ binding file
        cuda_sources=[open("gol_cuda_wideload.cpp").read()],   # contains both kernel and PYBIND11 module
        with_cuda=True,
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )

def gol_cuda_wideload(x: torch.Tensor, BLOCK_SIZE_ROW: int = None, BLOCK_SIZE_COL: int = None):
    if ext3 is None: init_ext3()
    if BLOCK_SIZE_ROW is None and BLOCK_SIZE_COL is None:
        BLOCK_SIZE_ROW = 4
        BLOCK_SIZE_COL = 512
    output = torch.empty_like(x)
    ext3.gol(x, output, BLOCK_SIZE_ROW, BLOCK_SIZE_COL)
    return output


# %%
ext4 = None 

def init_ext4():
    global ext4
    ext4 = load_inline(
        name="gol4_ext",
        cpp_sources="",            # no separate C++ binding file
        cuda_sources=[open("gol_cuda_grouped.cpp").read()],   # contains both kernel and PYBIND11 module
        with_cuda=True,
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )

def gol_cuda_grouped(x: torch.Tensor, BLOCK_SIZE_ROW: int = None, BLOCK_SIZE_COL: int = None):
    if ext4 is None: init_ext4()
    if BLOCK_SIZE_ROW is None and BLOCK_SIZE_COL is None:
        BLOCK_SIZE_ROW = 1
        BLOCK_SIZE_COL = 512
    output = torch.empty_like(x)
    ext4.gol(x, output, BLOCK_SIZE_ROW, BLOCK_SIZE_COL)
    return output


# %%
ext5 = None 

def init_ext5():
    global ext5
    ext5 = load_inline(
        name="gol5_ext",
        cpp_sources="",            # no separate C++ binding file
        cuda_sources=[open("gol_cuda_bitpacked.cpp").read()],   # contains both kernel and PYBIND11 module
        with_cuda=True,
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )

def gol_cuda_bitpacked(x: torch.Tensor, BLOCK_SIZE_ROW: int = None, BLOCK_SIZE_COL: int = None):
    if ext5 is None: init_ext5()
    if BLOCK_SIZE_ROW is None and BLOCK_SIZE_COL is None:
        BLOCK_SIZE_ROW = 4
        BLOCK_SIZE_COL = 32
    output = torch.empty_like(x)
    ext5.gol(x, output, BLOCK_SIZE_ROW, BLOCK_SIZE_COL)
    return output


# %%
ext6 = None 

def init_ext6():
    global ext6
    ext6 = load_inline(
        name="gol6_ext",
        cpp_sources="",            # no separate C++ binding file
        cuda_sources=[open("gol_cuda_bitpacked_64.cpp").read()],   # contains both kernel and PYBIND11 module
        with_cuda=True,
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )

def gol_cuda_bitpacked_64(x: torch.Tensor, BLOCK_SIZE_ROW: int = None, BLOCK_SIZE_COL: int = None):
    if ext6 is None: init_ext6()
    if BLOCK_SIZE_ROW is None and BLOCK_SIZE_COL is None:
        BLOCK_SIZE_ROW = 1
        BLOCK_SIZE_COL = 1024
    output = torch.empty_like(x)
    ext6.gol(x.view(torch.uint64), output.view(torch.uint64), BLOCK_SIZE_ROW, BLOCK_SIZE_COL)
    return output
    

# %%
ext7 = None 

def init_ext7():
    global ext7
    ext7 = load_inline(
        name="gol7_ext",
        cpp_sources="",            # no separate C++ binding file
        cuda_sources=[open("gol_cuda_grouped_bitpacked_64.cpp").read()],   # contains both kernel and PYBIND11 module
        with_cuda=True,
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )

def gol_cuda_grouped_bitpacked_64(x: torch.Tensor, BLOCK_SIZE_ROW: int = None, BLOCK_SIZE_COL: int = None):
    if ext7 is None: init_ext7()
    if BLOCK_SIZE_ROW is None and BLOCK_SIZE_COL is None:
        BLOCK_SIZE_ROW = 32
        BLOCK_SIZE_COL = 256
    output = torch.empty_like(x)
    ext7.gol(x.view(torch.uint64), output.view(torch.uint64), BLOCK_SIZE_ROW, BLOCK_SIZE_COL)
    return output


# %%
ext8 = None 

def init_ext8():
    global ext8
    ext8 = load_inline(
        name="gol8_ext",
        cpp_sources="",            # no separate C++ binding file
        cuda_sources=[open("gol_cuda_grouped_bitpacked_64_multistep.cpp").read()],   # contains both kernel and PYBIND11 module
        with_cuda=True,
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )

def gol_cuda_grouped_bitpacked_64_multistep(x: torch.Tensor, BLOCK_SIZE_ROW: int = None, BLOCK_SIZE_COL: int = None, STEPS: int = 4):
    if ext8 is None: init_ext8()
    if BLOCK_SIZE_ROW is None and BLOCK_SIZE_COL is None:
        BLOCK_SIZE_ROW = 32
        BLOCK_SIZE_COL = 256
    output = torch.empty_like(x)
    ext8.gol(x.view(torch.uint64), output.view(torch.uint64), BLOCK_SIZE_ROW, BLOCK_SIZE_COL, STEPS)
    return output
# %%
ext9 = None 

def init_ext9():
    global ext9
    ext9 = load_inline(
        name="gol9_ext",
        cpp_sources="",            # no separate C++ binding file
        cuda_sources=[open("gol_cuda_grouped_bitpacked_64_multistep_2.cpp").read()],   # contains both kernel and PYBIND11 module
        with_cuda=True,
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )

def gol_cuda_grouped_bitpacked_64_multistep_2(x: torch.Tensor, BLOCK_SIZE_ROW: int = None, BLOCK_SIZE_COL: int = None, STEPS: int = 4):
    if ext9 is None: init_ext9()
    if BLOCK_SIZE_ROW is None and BLOCK_SIZE_COL is None:
        BLOCK_SIZE_ROW = 32
        BLOCK_SIZE_COL = 256
    output = torch.empty_like(x)
    ext9.gol(x.view(torch.uint64), output.view(torch.uint64), BLOCK_SIZE_ROW, BLOCK_SIZE_COL, STEPS)
    return output

# %%
ext10 = None 

def init_ext10():
    global ext10
    ext10 = load_inline(
        name="gol10_ext",
        cpp_sources="",            # no separate C++ binding file
        cuda_sources=[open("gol_cuda_grouped_bitpacked_64_multistep_3.cpp").read()],   # contains both kernel and PYBIND11 module
        with_cuda=True,
        extra_cuda_cflags=["-O3"],
        # extra_cuda_cflags=["-O3 -G -lineinfo"],
        verbose=True,
    )

def gol_cuda_grouped_bitpacked_64_multistep_3(x: torch.Tensor, BLOCK_SIZE_ROW: int = None, BLOCK_SIZE_COL: int = None, STEPS: int = 4):
    if ext10 is None: init_ext10()
    if BLOCK_SIZE_ROW is None and BLOCK_SIZE_COL is None:
        BLOCK_SIZE_ROW = 32
        BLOCK_SIZE_COL = 256
    output = torch.empty_like(x)
    ext10.gol(x.view(torch.uint64), output.view(torch.uint64), BLOCK_SIZE_ROW, BLOCK_SIZE_COL, STEPS)
    return output