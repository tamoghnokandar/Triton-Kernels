import triton
import triton.language as tl

# Note: input, output are all float32 device tensors
@triton.jit
def relu_kernel(input_ptr, output_ptr, input_stride, output_stride, num_cols, block_size:tl.constexpr):
    pid = tl.program_id(axis=0)
    row_start = input_ptr + pid*input_stride
    col_offsets = tl.arange(0, block_size)
    row_idx = row_start + col_offsets
    mask = col_offsets < num_cols
    # Load the row to SRAM
    row = tl.load(row_idx, mask=mask, other=0.0)
    row = tl.where(row>0, row, 0)
    out_start = output_ptr + pid*output_stride
    out_idx = out_start + col_offsets
    tl.store(out_idx, row, mask=mask)
def solution(input, output, n: int, m: int):
    # assert input.shape == (m, n)
    # input = input.contiguous()
    # output = output.contiguous()
    n_rows, n_cols = input.shape
    block_size = triton.next_power_of_2(n_cols)
    grid = (n_rows,)
    relu_kernel[grid](input, output, input.stride(0), output.stride(0), n_cols, block_size)
    return output
  
