import os
import sys

import cutlass
import torch
import triton
import triton.language as tl


def get_area_bound(num_stages):
    """
    Return a reasonable upper bound on the block sizes dictated by the amount of shared memory.

    Found through manual testing at each num_stage.
    """
    if num_stages == 1:
        return 81920  # 2^{16} + 2^{14}
    elif num_stages == 2:
        return 49152  # 2^{15} + 2^{14}
    elif num_stages == 3:
        return 40960  # 2^{15} + 2^{13}
    elif num_stages == 4:
        return 24576  # 2^{14} + 2^{13}
    elif num_stages == 5:
        return 20480  # 2^{14} + 2^{12}
    elif num_stages == 6:
        return 16384  # 2^{13} + 2^{13}
    else:
        raise NotImplementedError


def get_benches():
    """
    Construct a list of benches to benchmark and plot using Triton's testing.perf_report function.
    """
    benches = [
        triton.testing.Benchmark(
            x_names=["K"],
            x_vals=[i for i in range(1024, 8193, 1024)],
            line_arg="provider",
            line_vals=["triton", "cublas", "cutlass"],
            line_names=["Triton", "cuBLAS", "cuTLASS"],
            styles=[("red", "-"), ("green", "-"), ("blue", "-")],
            ylabel="Time (ms)",
            plot_name=f"gsm{GSM}_k-autotuned_matmul_row-major_fp16",
            args={"M": 8192, "N": 8192, "GSM": GSM},
        )
        for GSM in [int(sys.argv[1])]  # single GSM, for job array
    ]
    assert benches, "Benches is empty"
    return benches


def get_configs():
    """
    Construct a list of configurations to be used for autotuning over (GROUP_SIZE_M, K).
    """
    configs = [
        triton.Config(
            {
                "BLOCK_SIZE_M": BSM,
                "BLOCK_SIZE_N": BSN,
                "BLOCK_SIZE_K": BSK,
            },
            num_stages=ns,
            num_warps=nw,
        )
        for BSM in [32, 64, 128, 256, 512]
        if BSM * int(sys.argv[1]) <= 8192
        for BSN in [32, 64, 128, 256, 512]
        for BSK in [32, 64, 128, 256, 512]
        for ns in [2, 3, 4, 5, 6]
        if (BSK * (BSM + BSN)) <= get_area_bound(ns)
        for nw in [4, 8, 16, 32]
    ]
    assert configs, "Configs is empty"
    return configs


@triton.autotune(configs=get_configs(), key=["K", "GROUP_SIZE_M"])
@triton.jit
def matmul_kernel(
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


def matmul(a, b, GROUP_SIZE_M):
    """
    Perform matrix multiplication using the provided matmul kernel.

    This is identical to the tutorial implementation, except that we allow for the option of passing in kernel
    parameters directly for later plotting (passing in GROUP_SIZE_M) purposes.
    """
    M, K = a.shape
    K, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)

    matmul_kernel[grid](
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


def run_benchmarks():
    """
    Plot the performance of the matmul kernel autotuned at every different (GROUP_SIZE_M, K).
    """

    @triton.testing.perf_report(get_benches())
    def benchmark(M, N, K, GSM, provider):
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16)

        if provider == "triton":
            print(f"gsm{GSM}_k{K}")
            mean_ms = triton.testing.do_bench(lambda: matmul(a, b, GSM))
        elif provider == "cublas":
            mean_ms = triton.testing.do_bench(lambda: torch.matmul(a, b))
        elif provider == "cutlass":
            plan = cutlass.op.Gemm(element=torch.float16, layout=cutlass.LayoutType.RowMajor)
            c = torch.empty((M, N), device="cuda", dtype=torch.float16)
            d = torch.empty((M, N), device="cuda", dtype=torch.float16)
            mean_ms = triton.testing.do_bench(lambda: plan.run(a, b, c, d))
        else:
            raise ValueError(f"Invalid provider: {provider}")

        return mean_ms

    benchmark.run(
        print_data=False,
        show_plots=False,
        save_path=os.path.join(output_dir, os.path.basename("gsm-k-autotuned_matmul_perf")),
    )


def main():
    os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

    # faster I/O to $SLURM_TMPDIR
    global output_dir
    output_dir = os.environ.get("SLURM_TMPDIR")
    assert output_dir, "$SLURM_TMPDIR does not exist"

    with open(os.path.join(output_dir, os.path.basename(f"gsm{int(sys.argv[1])}_autotuning.out")), "w") as sys.stdout:
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
        run_benchmarks()


if __name__ == "__main__":
    main()
