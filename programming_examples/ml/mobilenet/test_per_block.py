#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
"""End-to-end hardware test for one IRON per-block design.

Builds + runs a per-block xclbin (built from aie2_iron_per_block.py against the
bottleneck_A|B|C/data/ brevitas fixtures), and compares the actual NPU output
to the matching `golden_output_bn{N}_single.txt` brevitas reference.

Usage:
    python3 test_per_block.py --xclbin <path> --insts <path> --bn bn1 \\
        --fixture-dir bottleneck_A/data
"""

import argparse
import os
import sys
import numpy as np

import aie.iron as iron
from aie.utils.ml import DataShaper
from aie.utils import DefaultNPURuntime, NPUKernel

# (block_name, in_w, in_h, in_c, out_w, out_h, out_c, in_dtype, out_dtype)
BLOCK_SHAPES = {
    "bn1": (112, 112, 16, 56, 56, 24, np.int8, np.int8),
    "bn2": (56, 56, 24, 56, 56, 24, np.int8, np.int8),
    "bn3": (56, 56, 24, 28, 28, 40, np.int8, np.int8),
    "bn6": (28, 28, 40, 14, 14, 80, np.int8, np.int8),
    "bn7": (14, 14, 80, 14, 14, 80, np.int8, np.int8),
    "bn8": (14, 14, 80, 14, 14, 80, np.int8, np.int8),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xclbin", required=True)
    ap.add_argument("--insts", required=True)
    ap.add_argument("--bn", required=True, help="e.g. bn1")
    ap.add_argument(
        "--fixture-dir",
        required=True,
        help="path containing input_bn{N}_single.txt + golden_output_bn{N}_single.txt",
    )
    ap.add_argument("--atol", type=int, default=0)
    args = ap.parse_args()

    n = args.bn[2:]  # "bn1" -> "1"
    in_w, in_h, in_c, out_w, out_h, out_c, in_dtype, out_dtype = BLOCK_SHAPES[args.bn]
    vec = 8

    # ------------------------------------------------------------------
    # Load the xclbin and prepare buffers
    # ------------------------------------------------------------------
    npu_kernel = NPUKernel(args.xclbin, args.insts)
    kernel_handle = DefaultNPURuntime.load(npu_kernel)

    # Per-bn fixtures: int8 input flat (CHW), int8 golden flat (CHW).
    fix = args.fixture_dir
    raw_in = np.loadtxt(
        os.path.join(fix, f"input_bn{n}_single.txt"), delimiter=",", dtype=in_dtype
    )
    assert (
        raw_in.size == in_h * in_w * in_c
    ), f"input size {raw_in.size} != {in_h*in_w*in_c}"
    chw = raw_in.reshape(in_c, in_h, in_w)
    # AIE expects YCXC8 layout per-row (H, IC/8, W, 8).
    ds = DataShaper()
    ifm_mem_fmt = ds.reorder_mat(chw.astype(in_dtype), "YCXC8", "CYX")

    raw_gold = np.loadtxt(
        os.path.join(fix, f"golden_output_bn{n}_single.txt"),
        delimiter=",",
        dtype=out_dtype,
    )

    # ------------------------------------------------------------------
    # Set up input/output buffers and run
    # ------------------------------------------------------------------
    in_tensor = iron.tensor(ifm_mem_fmt.flatten().view(np.int32), dtype=np.int32)
    out_size_i32 = out_w * out_h * out_c // 4
    out_tensor = iron.zeros((out_size_i32,), dtype=np.int32)
    buffers = [in_tensor, out_tensor]

    print(f"  Running {args.bn} on NPU ...")
    DefaultNPURuntime.run(kernel_handle, buffers)

    # ------------------------------------------------------------------
    # Decode output: HCWC8 -> CHW, compare
    # ------------------------------------------------------------------
    aie_raw = out_tensor.numpy().view(out_dtype)
    aie_hcwc8 = aie_raw.reshape(out_h, out_c // vec, out_w, vec)
    aie_chw = ds.reorder_mat(aie_hcwc8, "CDYX", "YCXD").reshape(out_c, out_h, out_w)

    diff = aie_chw.astype(np.int32) - raw_gold.reshape(out_c, out_h, out_w).astype(
        np.int32
    )
    n_match = int((diff == 0).sum())
    n_total = aie_chw.size
    max_d = int(np.abs(diff).max())
    mean_d = float(np.abs(diff).mean())
    pct = 100.0 * n_match / n_total
    verdict = f"PASS_BN_E2E {args.bn}: {n_match}/{n_total} ({pct:.1f}%)  max={max_d}  mean={mean_d:.3f}"
    print(verdict)
    if max_d > args.atol:
        print(f"FAIL_BN_E2E {args.bn}: max diff {max_d} exceeds atol={args.atol}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
