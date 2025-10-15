import triton
import triton.language as tl

# Note: X, gamma, beta, Y are all float32 device tensors
@triton.autotune(
    configs=[
        # Add larger block sizes and more warps for large N
        triton.Config({'BLOCK_SIZE': 1024, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE': 2048, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE': 4096, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE': 8192, 'num_warps': 16}),
        triton.Config({'BLOCK_SIZE': 16384, 'num_warps': 16}),
    ],
    key=['N'],
)
@triton.jit
def layernorm_forward(X_ptr, Y_ptr, gamma_ptr, beta_ptr, stride_X, stride_Y, N, eps, BLOCK_SIZE:tl.constexpr):
    row = tl.program_id(axis=0)
    X_ptr += row*stride_X
    Y_ptr += row*stride_Y
    sum_accumulator = tl.zeros([BLOCK_SIZE],dtype=tl.float32)
    variance_accumultor = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols<N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0)
        sum_accumulator += x
    mean = tl.sum(sum_accumulator, axis=0)/N
    
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols<N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0)
        x_centered = tl.where(mask, x-mean, 0.0)
        variance_accumultor += x_centered*x_centered
    
    variance = tl.sum(variance_accumultor, axis=0)/N
    rstd = 1/tl.sqrt(variance+eps)
    # tl.store(mean_ptr+row, mean)
    # tl.store(rstd_ptr+row, rstd)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols<N
        gamma = tl.load(gamma_ptr + cols, mask=mask, other=0.0)
        beta = tl.load(beta_ptr + cols, mask=mask, other=0.0)
        x = tl.load(X_ptr + cols, mask=mask, other=0.0)
        x_normalized = (x-mean)*rstd
        y = x_normalized*gamma + beta
        tl.store(Y_ptr+cols,y,mask=mask)
    
    
def solution(X, gamma, beta, Y, B: int, F: int, D1: int, D2: int):
    MAX_FUSED_SIZE = 165536 // X.element_size()
    N = F*D1*D2
    # BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    # if N > BLOCK_SIZE:
    #     raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # BLOCK_SIZE, num_warps = calculate_settings(N)
    # heuristics for number of warps
    
    eps = 1e-5
    layernorm_forward[(B, )](X, Y, gamma, beta, X.stride(0), Y.stride(0), N, eps)
    return Y

  
