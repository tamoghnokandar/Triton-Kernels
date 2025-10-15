import triton
import triton.language as tl

# Note: X, Y are all float32 device tensors
@triton.jit
def rms_kernel(x_ptr, y_ptr, N, block_size:tl.constexpr):
    row = tl.program_id(axis=0)
    x_ptr += row*N
    y_ptr += row*N
    accumulator = tl.zeros([block_size], dtype=tl.float32)
    for offset in range(0, N, block_size):
        cols = offset + tl.arange(0, block_size)
        mask = cols<N
        x = tl.load(x_ptr+cols,mask=mask,other=0.0)
        accumulator += x*x
    mean = tl.sum(accumulator,axis=0)/N
    denominator = 1/tl.sqrt(mean+1e-5)
    for offset in range(0, N, block_size):
        cols = offset + tl.arange(0, block_size)
        mask = cols<N
        x = tl.load(x_ptr+cols,mask=mask,other=0.0)
        y = x*denominator
        y = tl.store(y_ptr+cols,y,mask=mask)
        
def solution(X, Y, B: int, N: int):
    block_size = triton.next_power_of_2(N)
    rms_kernel[(B,)](X, Y, N, block_size)
    return Y
  
