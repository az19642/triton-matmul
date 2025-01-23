import os
import sys

import cutlass
import torch
import triton
import triton.language as tl
import subprocess

MAX_BLOCK_SIZE_PROD = 2**23


@triton.jit
def base_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Triton tutorial implementation of the kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b, kernel, GROUP_SIZE_M):
    """
    Perform matrix multiplication using the provided matmul kernel.

    This is identical to the tutorial implementation, except that we allow for the option of passing in kernel
    parameters directly for later plotting (passing in GROUP_SIZE_M) purposes.
    """
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"

    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        GROUP_SIZE_M=GROUP_SIZE_M,
    )

    return c


def run_benchmarks(block_size_lst, gsm_lst, num_stages_lst, num_warps_lst) -> None:
    """
    Plot the performance of the matmul kernel autotuned at every different (GROUP_SIZE_M, K).
    """
    configs = []
    for bsm in block_size_lst:
        for bsn in block_size_lst:
            for bsk in block_size_lst:
                for ns in num_stages_lst:
                    for nw in num_warps_lst:
                        if bsm * bsn * bsk * (ns - 1) > MAX_BLOCK_SIZE_PROD:
                            continue
                        configs.append(
                            triton.Config(
                                {
                                    "BLOCK_SIZE_M": bsm,
                                    "BLOCK_SIZE_N": bsn,
                                    "BLOCK_SIZE_K": bsk,
                                },
                                num_stages=ns,
                                num_warps=nw,
                            )
                        )

    tunable_kernel = triton.autotune(configs=configs, key=["K", "GROUP_SIZE_M"])(
        base_kernel
    )

    benches = [
        triton.testing.Benchmark(
            x_names=["K"],
            x_vals=[i for i in range(512, 8193, 512)],
            line_arg="provider",
            line_vals=["triton", "cublas", "cutlass"],
            line_names=["Triton", "cuBLAS", "cuTLASS"],
            styles=[("red", "-"), ("green", "-"), ("blue", "-")],
            ylabel="Time (ms)",
            plot_name=f"gsm{gsm}_k-autotuned_matmul_row-major_fp16",
            args={
                "M": 8192,
                "N": 8192,
                "gsm": gsm,
            },
        )
        for gsm in gsm_lst
    ]

    @triton.testing.perf_report(benches)
    def benchmark(M, N, K, gsm, provider):
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16)

        plan = cutlass.op.Gemm(
            element=torch.float16, layout=cutlass.LayoutType.RowMajor
        )
        c = torch.empty((M, N), device="cuda", dtype=torch.float16)
        d = torch.empty((M, N), device="cuda", dtype=torch.float16)

        if provider == "triton":
            print(f"gsm{gsm}_k{K}")
            mean_ms = triton.testing.do_bench(lambda: matmul(a, b, tunable_kernel, gsm))
        elif provider == "cublas":
            mean_ms = triton.testing.do_bench(lambda: torch.matmul(a, b))
        elif provider == "cutlass":
            mean_ms = triton.testing.do_bench(lambda: plan.run(a, b, c, d))
        else:
            raise ValueError(f"Invalid provider: {provider}")

        return mean_ms

    benchmark.run(
        print_data=False,
        show_plots=False,
        save_path="./gsm-k-autotuned_matmul_perf",
    )


def main():
    os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

    # Lists of values for each parameter to grid tune over
    block_size_lst = [32, 64, 128, 256, 512, 1024]
    num_stages_lst = [2, 3]
    num_warps_lst = [8, 16, 32]
    gsm_list = [1, 2, 4, 8, 12, 16]
    with open("autotuning.out", "w") as sys.stdout:
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
        run_benchmarks(block_size_lst, gsm_list, num_stages_lst, num_warps_lst)


if __name__ == "__main__":
    main()
