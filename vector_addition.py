import triton
import triton.language as tl

# Note: d_input1, d_input2, d_output are all float32 device tensors
@triton.jit
def add_kernel(a_ptr, b_ptr, out_ptr, block_size:tl.constexpr, n_elements:tl.constexpr):
    pid = tl.program_id(axis=0)
    thread_start = pid*block_size
    offsets = thread_start + tl.arange(0, block_size)
    mask = offsets<n_elements
    a_pointer = tl.load(a_ptr+offsets, mask=mask) 
    b_pointer = tl.load(b_ptr+offsets, mask=mask) 
    tl.store(out_ptr+offsets, a_pointer+b_pointer,mask=mask)
def solution(d_input1, d_input2, d_output, n: int):
    block_size = 1024
    grid_size = triton.cdiv(n, block_size)
    grid = (grid_size, )
    add_kernel[grid](d_input1, d_input2, d_output, block_size, n)
    return d_output

  
