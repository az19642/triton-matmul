import collections
import os
import sys

import cutlass
import torch
import triton
import triton.language as tl


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


def matmul(a, b, kernel, kernel_params: dict):
    """
    Perform matrix multiplication using the provided matmul kernel.

    This is identical to the tutorial implementation, except that we allow for the option of passing in kernel
    parameters directly for later plotting (passing in GROUP_SIZE_M) purposes.

    :param a: input matrix A
    :param b: input matrix B
    :param kernel: the Triton matmul kernel to use
    :param kernel_params: kernel parameters
    """
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"

    # kernel_opts is used for plotting purposes (e.g, pass in GROUP_SIZE_M directly to matmul instead of to the matmul kernel through the autotuner)

    M, K = a.shape
    K, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)

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
        **kernel_params,
    )

    return c


def estimate_optimal_conf() -> int:
    """
    Estimate the optimal configuration for matmul by autotuning with a subset of the K values (0, 8193).

    The autotuning process will produce an optimal configuration for each K in the subset.
    We do a greedy estimate by choosing the most frequent configuration across all K.
    The variance of our estimate decreases as we increase the subset size, among other factors.

    :return: the number of configurations used in the autotuning process
    """
    configs = []
    block_size_lst = [32, 64, 128, 256, 512]
    gsm_lst = [1, 2, 4, 8, 10, 12, 14]  # GROUP_SIZE_M
    ns_lst = [1, 2, 3, 4]  # num_stages
    nw_lst = [4, 8, 16, 32]  # num_warps
    for bsm in block_size_lst:
        for bsn in block_size_lst:
            for bsk in block_size_lst:
                for gsm in gsm_lst:
                    for ns in ns_lst:
                        for nw in nw_lst:
                            configs.append(
                                triton.Config(
                                    {
                                        "BLOCK_SIZE_M": bsm,
                                        "BLOCK_SIZE_N": bsn,
                                        "BLOCK_SIZE_K": bsk,
                                        "GROUP_SIZE_M": gsm,
                                    },
                                    num_stages=ns,
                                    num_warps=nw,
                                )
                            )

    # set kernel to be autotunable with the above configs
    auto_kernel = triton.autotune(configs=configs, key=["K"])(base_kernel)

    bench = triton.testing.Benchmark(
        x_names=["K"],
        x_vals=[i for i in range(256, 8193, 256)],  # note that len(x_vals) = 16
        line_arg="provider",
        line_vals=["triton", "cublas", "cutlass"],
        line_names=["Triton", "cuBLAS", "cuTLASS"],
        styles=[("red", "-"), ("blue", "-"), ("green", "-")],
        ylabel="Mean runtime (ms)",
        plot_name=f"per-k-autotuned_matmul_row-major_fp16",
        args={"M": 8192, "N": 8192},
    )

    @triton.testing.perf_report(bench)
    def benchmark(M, N, K, provider):
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16)

        plan = cutlass.op.Gemm(element=torch.float16, layout=cutlass.LayoutType.RowMajor)
        # c stores the output of matmul(a, b) and d is a dummy tensor
        c = torch.ones((M, N), device="cuda", dtype=torch.float16)
        d = torch.ones((M, N), device="cuda", dtype=torch.float16)

        if provider == "cublas":
            mean_ms = triton.testing.do_bench(lambda: torch.matmul(a, b))
        elif provider == "triton":
            mean_ms = triton.testing.do_bench(lambda: matmul(a, b, auto_kernel, {}))
        elif provider == "cutlass":
            mean_ms = triton.testing.do_bench(lambda: plan.run(a, b, c, d))

        return mean_ms

    benchmark.run(print_data=True, show_plots=True, save_path="./per-k-autotuned_matmul_perf")
    return len(configs)


def extract_config(line: str) -> tuple[dict, dict]:
    """
    Extract the meta and compilation parameters from a line in the autotuning output.

    :param line: a line in the autotuning output
    :return: two dictionaries containing the meta and compilation parameters
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


def get_most_freq_config(file_path: str, num_configs: int) -> tuple[triton.Config, int]:
    """
    Get the most frequent configuration from the autotuning output.

    :param file_path: the path to the autotuning output
    :param num_configs: the number of configurations contained in the file
    :return: the most frequent Triton configuration and the GROUP_SIZE_M value (separately)
    """
    config_counter = collections.Counter()
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            if i >= num_configs:
                break
            config_counter[line] += 1

    most_freq, _ = config_counter.most_common(1)[0]

    print(f"Most frequent config: {most_freq}")

    meta_dct, comp_dct = extract_config(most_freq)
    gsm = meta_dct["GROUP_SIZE_M"]
    del meta_dct["GROUP_SIZE_M"]

    return triton.Config(meta_dct, **comp_dct), gsm


def plot_near_optimal(optimal_conf: triton.Config, optimal_gsm) -> None:
    """
    Plot the performance of the matmul kernel for different GROUP_SIZE_M values.

    The plots will include the optimal GROUP_SIZE_M found in the autotuning process; the intent is to show the
    behaviour of the kernel near the (estimated) optimal GROUP_SIZE_M.

    :param optimal_conf: the optimal configuration found in the autotuning process, excludes GROUP_SIZE_M
    """

    # We set the configuration of the kernel using an autotuner with one config; Triton does not seem to provide another way
    optimal_kernel = triton.autotune(configs=[optimal_conf], key=["M", "N"])(base_kernel)  # static

    benches = [
        triton.testing.Benchmark(
            x_names=["K"],
            x_vals=[i for i in range(256, 8193, 256)],
            line_arg="provider",
            line_vals=["triton", "cublas", "cutlass"],
            line_names=["Triton", "cuBLAS", "cuTLASS"],
            styles=[("red", "-"), ("green", "-"), ("blue", "-")],
            ylabel="Time (ms)",
            plot_name=f"GSM{GSM}_autotuned_matmul_row-major_fp16",
            args={
                "M": 8192,
                "N": 8192,
                "GSM": GSM,
            },
        )
        for GSM in list({optimal_gsm - 2 * i for i in range(4)} | {optimal_gsm + 2 * i for i in range(4)})
    ]

    @triton.testing.perf_report(benches)
    def benchmark(M, N, K, GSM, provider):
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16)

        plan = cutlass.op.Gemm(element=torch.float16, layout=cutlass.LayoutType.RowMajor)
        # c stores the output of matmul(a, b) and d is a dummy tensor
        c = torch.ones((M, N), device="cuda", dtype=torch.float16)
        d = torch.ones((M, N), device="cuda", dtype=torch.float16)

        if provider == "cublas":
            mean_ms = triton.testing.do_bench(lambda: torch.matmul(a, b))
        elif provider == "triton":
            mean_ms = triton.testing.do_bench(lambda: matmul(a, b, optimal_kernel, {"GROUP_SIZE_M": GSM}))
        elif provider == "cutlass":
            mean_ms = triton.testing.do_bench(lambda: plan.run(a, b, c, d))

        return mean_ms

    benchmark.run(print_data=True, show_plots=True, save_path="./autotuned_matmul_perf")


def main():
    stdout = sys.stdout
    with open("autotuning_output", "w") as sys.stdout:
        os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
        num_configs = estimate_optimal_conf()

    sys.stdout = stdout  # reset stdout

    optimal_config, optimal_gsm = get_most_freq_config("autotuning_output", num_configs)

    os.environ["MLIR_ENABLE_DUMP"] = "1"
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    os.environ["LLVM_IR_ENABLE_DUMP"] = "1"
    plot_near_optimal(optimal_config, optimal_gsm)


if __name__ == "__main__":
    main()
