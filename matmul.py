import triton
import triton.language as tl

# Note: input_a, input_b, output_c are all float32 device tensors
@triton.jit
def matmul_kernel(a_ptr, b_ptr, out_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, block_size_m:tl.constexpr, block_size_n:tl.constexpr, block_size_k:tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offset_m = pid_m*block_size_m + tl.arange(0, block_size_m) 
    offset_n = pid_n*block_size_n + tl.arange(0, block_size_n)
    acc = tl.zeros((block_size_m, block_size_n), dtype=tl.float32)
    for k in range(0, K, block_size_k):
        offset_k = k + tl.arange(0, block_size_k)
        a_ptrs = a_ptr + offset_m[:,None]*stride_am + offset_k[None,:]*stride_ak
        b_ptrs = b_ptr + offset_k[:,None]*stride_bk + offset_n[None,:]*stride_bn
        a = tl.load(a_ptrs, mask=(offset_m[:,None]<M) & (offset_k[None,:]<K), other=0.0)
        b = tl.load(b_ptrs, mask=(offset_n[None,:]<N) & (offset_k[:,None]<K), other=0.0)
        acc += tl.dot(a, b)
    out_ptrs = out_ptr + offset_m[:,None]*stride_cm + offset_n[None,:]*stride_cn
    tl.store(out_ptrs, acc, mask=(offset_m[:,None]<M) & (offset_n[None,:]<N))

def solution(input_a, input_b, output_c, m: int, n: int, k: int):
    block_size_k = 32
    block_size_m = block_size_n = 64
    grid = (triton.cdiv(m,block_size_m),triton.cdiv(n,block_size_n))
    matmul_kernel[grid](input_a,input_b,output_c,m,n,k,input_a.stride(0),input_a.stride(1),input_b.stride(0),input_b.stride(1),output_c.stride(0),output_c.stride(1),block_size_m,block_size_n,block_size_k)
    return output_c

  
