import collections
import os
import sys

import cutlass
import torch
import triton
import triton.language as tl


def get_configs():
    """
    Return a list of configurations to autotune over K.
    """
    MAX_BLOCK_SIZE_PROD = 2**23
    configs = [
        triton.Config(
            {
                "BLOCK_SIZE_M": BSM,
                "BLOCK_SIZE_N": BSN,
                "BLOCK_SIZE_K": BSK,
                "GROUP_SIZE_M": GSM,
            },
            num_stages=ns,
            num_warps=nw,
        )
        # for BSM in [32, 64, 128, 256]
        # for BSN in [32, 64, 128, 256]
        # for BSK in [32, 64, 128, 256]
        # for GSM in [1, 2, 4, 8, 12, 16, 20, 32, 48, 62]
        # for ns in [2, 3]
        # for nw in [8, 16, 32]
        for BSM in [32, 64]
        for BSN in [32, 64]
        for BSK in [32, 64]
        for GSM in [2, 4, 8]
        for ns in [2]
        for nw in [32]
        if BSM * BSN * BSK * (ns - 1) <= MAX_BLOCK_SIZE_PROD
    ]
    assert configs, "Configs is empty"
    global num_configs
    num_configs = len(configs)
    return configs


def get_benches():
    """
    Return a list of benches to benchmark the optimal config and plot using Triton's testing.perf_report function.
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
            plot_name=f"GSM{GSM}_optimal_matmul_row-major-fp16",
            args={"M": 8192, "N": 8192, "GSM": GSM},
        )
        for GSM in [1, 2, 4, 8, 12, 16, 20, 32, 48, 62]
    ]
    assert benches, "Benches is empty"
    return benches


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
    """Triton tutorial implementation of the kernel for computing the matmul C = A x B.
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


def matmul(a, b, kernel, GROUP_SIZE_M=None):
    """
    Perform matrix multiplication using the provided matmul kernel.

    This is identical to the tutorial implementation, except that we allow for the option of passing in kernel
    parameters directly for later plotting (passing in GROUP_SIZE_M) purposes.
    """
    M, K = a.shape
    K, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
    kernel_args = [
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
    ]
    if GROUP_SIZE_M:
        kernel_args.append(GROUP_SIZE_M)
    kernel[grid](*kernel_args)

    return c


def get_optimal_config() -> int:
    """
    Estimate the optimal configuration for matmul by autotuning with key=K
    """

    # set kernel to be autotunable with the above configs
    tunable_kernel = triton.autotune(configs=get_configs(), key=["K"])(triton.jit()(matmul_kernel))

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["K"],
            x_vals=[i for i in range(1024, 8193, 1024)],
            line_arg="provider",
            line_vals=["triton", "cublas", "cutlass"],
            line_names=["Triton", "cuBLAS", "cuTLASS"],
            styles=[("red", "-"), ("blue", "-"), ("green", "-")],
            ylabel="Mean runtime (ms)",
            plot_name=f"per-k-autotuned_matmul_row-major_fp16",
            args={"M": 8192, "N": 8192},
        )
    )
    def benchmark(M, N, K, provider):
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16)

        if provider == "triton":
            print(f"k{K}")
            mean_ms = triton.testing.do_bench(lambda: matmul(a, b, tunable_kernel))
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
        save_path=os.path.join(output_dir, os.path.basename("k-autotuned_matmul_perf")),
    )


def _extract_config(line: str) -> tuple[dict, dict]:
    """
    Extract the meta and compilation parameters from a line in the autotuning output.
    """
    # find the start and end of the meta parameters
    meta_start = line.find("BLOCK_SIZE_M")
    meta_end = line.find("num_warps") - 2

    # find the start and end of the compilation parameters
    comp_start = meta_end + 2
    comp_end = len(line) - 2

    meta = line[meta_start:meta_end].split(", ")
    comp = line[comp_start:comp_end].split(", ")

    meta_dct = {}
    for param_val in meta:
        param, val = param_val.split(": ")
        meta_dct[param] = int(val)

    comp_dct = {}
    for param_val in comp:
        param, val = param_val.split(": ")
        comp_dct[param] = None if val == "None" else int(val)

    return meta_dct, comp_dct


def get_majority_config(file_path: str) -> tuple[triton.Config, int]:
    """
    Get the most frequent configuration from the autotuning output.
    """
    config_counter = collections.Counter()
    count_configs = 0
    with open(file_path, "r") as f:
        for line in f:
            if count_configs > num_configs or line[0] == "k":
                continue
            config_counter[line] += 1
            count_configs += 1

    most_freq, _ = config_counter.most_common(1)[0]

    print(f"Most frequent config: {most_freq}")

    meta_dct, comp_dct = _extract_config(most_freq)
    gsm = meta_dct["GROUP_SIZE_M"]
    del meta_dct["GROUP_SIZE_M"]

    return triton.Config(meta_dct, **comp_dct), gsm


def plot_near_optimal(optimal_config: triton.Config) -> None:
    """
    Plot the performance of the matmul kernel for different GROUP_SIZE_M values.

    The plots will include the optimal GROUP_SIZE_M found in the autotuning process; the intent is to show the
    effects of GROUP_SIZE_M near the optimal GROUP_SIZE_M.
    """

    # We set the configuration of the kernel using a static (key) autotuner; Triton does not seem to provide another way
    optimal_kernel = triton.autotune(configs=[optimal_config], key=["M", "N"])(triton.jit()(matmul_kernel))

    @triton.testing.perf_report(get_benches())
    def benchmark(
        M,
        N,
        K,
        GSM,
        provider,
    ):
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16)

        if provider == "triton":
            mean_ms = triton.testing.do_bench(lambda: matmul(a, b, optimal_kernel, GSM))
        elif provider == "cublas":
            mean_ms = triton.testing.do_bench(lambda: torch.matmul(a, b))
        elif provider == "cutlass":
            c = torch.empty((M, N), device="cuda", dtype=torch.float16)
            d = torch.empty((M, N), device="cuda", dtype=torch.float16)
            plan = cutlass.op.Gemm(element=torch.float16, layout=cutlass.LayoutType.RowMajor)
            mean_ms = triton.testing.do_bench(lambda: plan.run(a, b, c, d))
        else:
            raise ValueError(f"Invalid provider: {provider}")

        return mean_ms

    benchmark.run(
        print_data=True, show_plots=True, save_path=os.path.join(output_dir, os.path.basename("optimal_matmul_perf"))
    )


def main():
    os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
    global output_dir
    output_dir = os.environ.get("SLURM_TMPDIR")
    autotuning_path = os.path.join(output_dir, os.path.basename("autotuning.out"))

    stdout = sys.stdout
    with open(autotuning_path, "w") as sys.stdout:
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
        optimal_config = get_optimal_config()
    sys.stdout = stdout

    # os.environ["TRITON_PRINT_AUTOTUNING"] = "0"
    # os.environ["MLIR_ENABLE_DUMP"] = "1"
    # os.environ["LLVM_IR_ENABLE_DUMP"] = "1"
    # os.environ["MLIR_DUMP_PATH"] = "dump.out"
    optimal_config, _ = get_majority_config(autotuning_path)
    plot_near_optimal(optimal_config)


if __name__ == "__main__":
    main()
