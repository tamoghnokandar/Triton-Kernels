import triton
import triton.language as tl

# Note: input, output are all float32 device tensors
@triton.jit
def sigmoid_kernel(input_ptr, output_ptr, n_cols, block_size:tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid*n_cols + tl.arange(0, block_size)
    mask = tl.arange(0, block_size) < n_cols
    x = tl.load(input_ptr+offsets,mask=mask)
    tl.store(output_ptr+offsets,tl.sigmoid(x),mask=mask)
def solution(input, output, n: int, m: int):
    n_rows, n_cols = input.shape
    grid = (n_rows, )
    block_size = triton.next_power_of_2(n_cols)
    sigmoid_kernel[grid](input, output, n_cols, block_size)
    return output

  
