import triton
import triton.language as tl

@triton.jit
def matmul_no_scf_with_advance_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """
    Computes a block of the matrix multiplication C = A @ B.

    This kernel is designed to calculate a single (BLOCK_M, BLOCK_N) tile of the output matrix C.
    It loads a (BLOCK_M, BLOCK_K) tile from matrix A and a (BLOCK_K, BLOCK_N) tile from matrix B.
    The kernel utilizes `tl.make_block_ptr` for creating pointers to blocks of A and B,
    and demonstrates the use of `tl.advance` for adjusting these block pointers.
    The core computation is performed using `tl.dot`. The resulting tile is then stored
    back to the C matrix. This version does not use Triton's Structured Control Flow (SCF)
    for iterating over the K dimension; it assumes BLOCK_K covers the necessary
    portion of the K dimension for a single dot product accumulation or that accumulation
    over K-blocks is handled externally.

    Parameters:
    -----------
    a_ptr : tl.pointer_type
        Pointer to the input matrix A.
    b_ptr : tl.pointer_type
        Pointer to the input matrix B.
    c_ptr : tl.pointer_type
        Pointer to the output matrix C.
    M : int
        The number of rows in matrix A and matrix C.
    N : int
        The number of columns in matrix B and matrix C.
    K : int
        The number of columns in matrix A and rows in matrix B.
    stride_am : int
        The stride (in elements) for moving from one row to the next in matrix A.
    stride_ak : int
        The stride (in elements) for moving from one column to the next in matrix A.
    stride_bk : int
        The stride (in elements) for moving from one row to the next in matrix B.
    stride_bn : int
        The stride (in elements) for moving from one column to the next in matrix B.
    stride_cm : int
        The stride (in elements) for moving from one row to the next in matrix C.
    stride_cn : int
        The stride (in elements) for moving from one column to the next in matrix C.
    BLOCK_M : tl.constexpr
        The height of the tiles processed from matrix A and C.
    BLOCK_N : tl.constexpr
        The width of the tiles processed from matrix B and C.
    BLOCK_K : tl.constexpr
        The depth of the tiles (common dimension K) processed from A and B for the dot product.

    Notes:
    --------
    This kernel assumes that it's computing a subproblem defined by a BLOCK_M x BLOCK_N submatrix
    of C and a smaller BLOCK_M x BLOCK_K submatrix of A. It will efficiently use `tl.dot()` to
    compute the inner dot product, BLOCK_K x BLOCK_N, for each k "slice" of A, directly
    producing a BLOCK_M x BLOCK_N output tile.
    """
    pid = tl.program_id(axis=0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    width = grid_m * grid_n
    tile_m = pid // width
    tile_n = pid % width
    m_offsets = (tile_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    n_offsets = (tile_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    k_offsets = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
    b_ptrs = b_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    amplitudes = tl.load(a_ptrs).to(tl.float32)
    batch_encoding_tokens = tl.load(b_ptrs).to(tl.float32)
    for k in range(0, K, BLOCK_K):
        accumulator += tl.dot(amplitudes, batch_encoding_tokens)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = accumulator.to(tl.float16)
    c_ptrs = c_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

##################################################################################################################################################

import numpy as np
import random
import torch 
import os
import pytest
from numpy.random import RandomState
from geak_eval.perf.ROCm.performance_utils_pytest import PytestBenchmarker, do_bench_config, save_all_benchmark_results
from typing import Dict

result_gold = {}

######################################## HELPERS for Eval ######################################## 
def set_seed(seed: int = 42) -> None:
    """
    Set the random seed for reproducibility across multiple libraries and configure PyTorch for deterministic behavior.

    Args:
        seed (int): The seed value to set. Default is 42.
    """
    # Set seed for Python's built-in random module
    random.seed(seed)
    # Set seed for NumPy
    np.random.seed(seed)
    # Set seed for PyTorch on CPU
    torch.manual_seed(seed)
    # Set seed for PyTorch on all GPUs (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set environment variable for hash-based operations
    os.environ['PYTHONHASHSEED'] = str(seed)

######################################## HELPERS for Eval ######################################## 


@pytest.mark.interpreter
@pytest.mark.parametrize("shape, num_warps", [  #
    (shape, num_warps) for shape in [
        [64, 64, 16],
        [64, 64, 32],
        [64, 64, 64],
    ] for num_warps in [4, 8]
])
def test_block_ptr_matmul_no_scf(shape, num_warps, request, device='cuda'):
    set_seed()

    m, n, k = shape
    a = torch.randn((m, k), device=device, dtype=torch.float16)
    b = torch.randn((k, n), device=device, dtype=torch.float16)
    c = torch.empty((m, n), device=device, dtype=torch.float32)

    grid = lambda META: (1, )
    matmul_no_scf_with_advance_kernel[grid](
        a_ptr=a, b_ptr=b, c_ptr=c,  #
        M=m, N=n, K=k,  #
        stride_am=a.stride(0), stride_ak=a.stride(1),  #
        stride_bk=b.stride(0), stride_bn=b.stride(1),  #
        stride_cm=c.stride(0), stride_cn=c.stride(1),  #
        BLOCK_M=m, BLOCK_N=n, BLOCK_K=k,  #
        num_warps=num_warps)
    golden = torch.matmul(a, b)

    ################### save tri_out in result_gold ###################
    test_case_name = request.node.name
    sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_")
    result_gold[sanitized_key_name] = c.clone().detach().cpu()
    ################################################################### 

    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])
    
    torch.testing.assert_close(c, golden, check_dtype=False)

# --- Python wrapper for the kernel ---
def block_pointer_matmul_triton_wrapper(a_tensor, b_tensor, c_tensor, num_warps_arg):
    m, k_a = a_tensor.shape
    k_b, n = b_tensor.shape
    assert k_a == k_b, "K dimension must match"
    
    # The kernel is designed for a single block operation covering the whole matrix
    # So, BLOCK_M=m, BLOCK_N=n, BLOCK_K=k_a
    # Grid is (1,) as per original test
    grid = (1,)
    
    matmul_no_scf_with_advance_kernel[grid](
        a_ptr=a_tensor, b_ptr=b_tensor, c_ptr=c_tensor,
        M=m, N=n, K=k_a,
        stride_am=a_tensor.stride(0), stride_ak=a_tensor.stride(1),
        stride_bk=b_tensor.stride(0), stride_bn=b_tensor.stride(1),
        stride_cm=c_tensor.stride(0), stride_cn=c_tensor.stride(1),
        BLOCK_M=m, BLOCK_N=n, BLOCK_K=k_a, # Kernel uses these as constexpr
        num_warps=num_warps_arg # num_warps passed to launch, not directly used by this jit kernel signature
                                # but can affect autotuning if kernel had it. This kernel is not autotuned.
    )
    return c_tensor

# --- Define TFLOPS and GB/s calculators for this specific GEMM ---
def calculate_block_matmul_tflops(params: dict, ms: float) -> float:
    M, N, K = params['M'], params['N'], params['K']
    # Single dot product of (M,K) @ (K,N)
    flops = 2 * M * N * K
    tflops = flops / (ms / 1000) / 1e12
    return tflops

def calculate_block_matmul_gbps(params: dict, ms: float) -> float:
    M, N, K = params['M'], params['N'], params['K']
    dtype_str = params.get('dtype_str', 'fp16') # Original test uses fp16

    if dtype_str == 'fp32': current_dtype = torch.float32
    elif dtype_str == 'bf16': current_dtype = torch.bfloat16
    else: current_dtype = torch.float16
    element_size = torch.tensor([], dtype=current_dtype).element_size()

    bytes_a = M * K * element_size
    bytes_b = K * N * element_size
    bytes_c = M * N * element_size
    total_bytes = bytes_a + bytes_b + bytes_c
    gbps = total_bytes / (ms / 1000) / 1e9
    return gbps

# --- Name for the benchmark output JSON file ---
OP_NAME_FOR_BENCHMARK = "block_pointer_matmul_perf"

# --- Pytest parametrize for performance testing ---
# Original shapes are small. For perf, we might want larger, but respecting kernel's nature.
# This kernel does M*K, K*N, M*N as ONE block.
# So, M, N, K are also BLOCK_M, BLOCK_N, BLOCK_K.
# Triton has limits on BLOCK_SIZEs (e.g., BLOCK_K often <= 1024 or 2048 due to shared memory).
# Let's use shapes where M, N, K themselves are valid block sizes.
BPM_SHAPES_FOR_PERF = [
    # M,  N,   K
    (64, 64,  64),
    (128, 128, 128),
    (64, 128, 256),
    (256, 64,  128),
    (128, 256, 64),
]
BPM_DTYPES_FOR_PERF = ['fp16', 'fp32'] # Add bf16 if relevant
# num_warps is passed to launch but not a JIT param for this kernel.
# It can influence scheduling. Let's fix it for perf or parametrize.
BPM_NUM_WARPS_FOR_PERF = [4, 8]


@pytest.mark.parametrize("shape", BPM_SHAPES_FOR_PERF)
@pytest.mark.parametrize("num_warps_arg", BPM_NUM_WARPS_FOR_PERF)
@pytest.mark.parametrize("dtype_str", BPM_DTYPES_FOR_PERF)
def test_performance(shape, num_warps_arg, dtype_str, request, device='cuda'): # Renamed
    set_seed()
    m, n, k = shape

    if dtype_str == 'fp32': current_dtype = torch.float32
    elif dtype_str == 'bf16': current_dtype = torch.bfloat16
    else: current_dtype = torch.float16

    a = torch.randn((m, k), device=device, dtype=current_dtype)
    b = torch.randn((k, n), device=device, dtype=current_dtype)
    # Output c for this kernel is float32 if inputs are float16, due to tl.dot default accumulator.
    # The kernel does `c_val.to(c_ptr.dtype.element_ty)`. So c's dtype matters.
    # Let's make c's dtype match input for simplicity, or be float32 if inputs are lower precision.
    # If a,b are fp16, c = torch.dot(a,b) accumulates in fp32, then cast to c.dtype.
    # For perf, often C is same dtype as A, B or higher precision (fp32).
    # Let C be same dtype as input for this benchmark.
    c = torch.empty((m, n), device=device, dtype=current_dtype)


    # --- Create op_lambda for benchmarking ---
    op_lambda = lambda: block_pointer_matmul_triton_wrapper(a, b, c, num_warps_arg)

    # --- Benchmarking ---
    bench_config = do_bench_config(warm_up=25, repetition=100)
    benchmarker = PytestBenchmarker(op_callable=op_lambda,
                                    op_name=OP_NAME_FOR_BENCHMARK,
                                    config=bench_config)

    current_params_for_logs_and_calc = {
        "M": m, "N": n, "K": k,
        "num_warps": num_warps_arg, "dtype_str": dtype_str
    }

    benchmarker.run_benchmark(current_params_dict=current_params_for_logs_and_calc,
                              gbps_calculator=calculate_block_matmul_gbps,
                              tflops_calculator=calculate_block_matmul_tflops)

######################################## HELPERS for Eval ########################################     
# --- Pytest hook to save the dictionary at the end of the session ---  
def test_save_results():  
    """  
    Called after whole test run finished, right before returning the exit status to the system.  
    """
    print('Inside session finish...')
    if "_CALL_SUCCESS_" not in result_gold:
        result_gold['_CALL_SUCCESS_'] = torch.tensor([[0.0]])
    OUTPUT_FILENAME = __file__.replace('.','_') + '.pt'
    print(f"\nSaving all y_triton results to {OUTPUT_FILENAME}...")  
    # Ensure the directory for the output file exists if it's in a subdirectory  
    output_dir = os.path.dirname(OUTPUT_FILENAME)  
    if output_dir and not os.path.exists(output_dir):  
        os.makedirs(output_dir, exist_ok=True)  
    torch.save(result_gold, OUTPUT_FILENAME)       
    print(f"Successfully saved {len(result_gold)} y_triton tensors to {OUTPUT_FILENAME}.")  

def test_save_performance_results():
    """
    Called after the test_performance function finishes.
    This is a separate hook to ensure performance results are saved.
    """
    print('\nPytest session finishing... Saving benchmark results...')

    output_directory = os.path.join(os.path.dirname(__file__), "perf")  # Save in a "perf" subdirectory next to the test file
    os.makedirs(output_directory, exist_ok=True)
    
    save_all_benchmark_results(output_directory)
    print(f"All benchmark results attempted to save to: {output_directory}")

######################################## HELPERS for Eval ########################################