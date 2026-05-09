#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
"""End-to-end hardware test for an IRON chain design (bn10..bn12 or bn13..bn14).

Loads the per-chain brevitas fixtures from bottleneck_B|C/data/, runs the
chain xclbin on the NPU, and compares bit-exact against golden_output.txt.

Cascade chains additionally upload the 4 cascade weight buffers (bn13_l1,
bn13_l3, bn14_l1, bn14_l3) to the AIE via shim DMA.

Usage:
    python3 test_chain.py --mode pipeline --xclbin <p> --insts <p> \\
        --fixture-dir bottleneck_B/data
    python3 test_chain.py --mode cascade --xclbin <p> --insts <p> \\
        --fixture-dir bottleneck_C/data
"""

import argparse
import os
import sys
import numpy as np

import aie.iron as iron
from aie.utils.ml import DataShaper
from aie.utils import DefaultNPURuntime, NPUKernel

# (in_w, in_h, in_c, out_w, out_h, out_c)
SHAPES = {
    "pipeline": (14, 14, 80, 7, 7, 80),
    "cascade": (7, 7, 80, 7, 7, 80),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["pipeline", "cascade"])
    ap.add_argument("--xclbin", required=True)
    ap.add_argument("--insts", required=True)
    ap.add_argument("--fixture-dir", required=True)
    ap.add_argument("--atol", type=int, default=0)
    args = ap.parse_args()

    in_w, in_h, in_c, out_w, out_h, out_c = SHAPES[args.mode]
    vec = 8
    fix = args.fixture_dir.rstrip("/") + "/"

    npu_kernel = NPUKernel(args.xclbin, args.insts)
    kernel_handle = DefaultNPURuntime.load(npu_kernel)
    ds = DataShaper()

    # --- input (CHW int8) -> YCXC8 ---
    raw_in = np.loadtxt(
        fix + "before_ifm_mem_fmt_1x1.txt", delimiter=",", dtype=np.int8
    )
    assert (
        raw_in.size == in_h * in_w * in_c
    ), f"input size {raw_in.size} != {in_h*in_w*in_c}"
    chw = raw_in.reshape(in_c, in_h, in_w)
    ifm_mem_fmt = ds.reorder_mat(chw, "YCXC8", "CYX")
    in_tensor = iron.tensor(ifm_mem_fmt.flatten().view(np.int32), dtype=np.int32)

    # --- output buffer (i32 view of HCWC8 int8) ---
    out_size_i32 = out_w * out_h * out_c // 4
    out_tensor = iron.zeros((out_size_i32,), dtype=np.int32)

    # --- cascade weights: ONE concatenated buffer (matches aie2_iron_chain.py
    #     and main mobilenet's single-cascade-weight-tensor pattern). Order:
    #     bn13_l1 | bn13_l3(put+get) | bn14_l1 | bn14_l3(put+get).
    buffers = [in_tensor]
    if args.mode == "cascade":
        bn13_l1 = np.loadtxt(fix + "bn13_1_chain.txt", delimiter=",", dtype=np.int8)
        bn13_l3 = np.concatenate(
            [
                np.loadtxt(fix + "bn13_3_put_chain.txt", delimiter=",", dtype=np.int8),
                np.loadtxt(fix + "bn13_3_get_chain.txt", delimiter=",", dtype=np.int8),
            ]
        )
        bn14_l1 = np.loadtxt(fix + "bn14_1_chain.txt", delimiter=",", dtype=np.int8)
        bn14_l3 = np.concatenate(
            [
                np.loadtxt(fix + "bn14_3_put_chain.txt", delimiter=",", dtype=np.int8),
                np.loadtxt(fix + "bn14_3_get_chain.txt", delimiter=",", dtype=np.int8),
            ]
        )
        all_wts = np.concatenate([bn13_l1, bn13_l3, bn14_l1, bn14_l3])
        assert all_wts.size == 4 * 80 * 960
        buffers.append(iron.tensor(all_wts.view(np.int32), dtype=np.int32))
    buffers.append(out_tensor)

    print(f"  Running {args.mode} chain on NPU ...")
    DefaultNPURuntime.run(kernel_handle, buffers)

    # --- decode output: HCWC8 -> CHW, compare ---
    aie_raw = out_tensor.numpy().view(np.int8)
    aie_hcwc8 = aie_raw.reshape(out_h, out_c // vec, out_w, vec)
    aie_chw = ds.reorder_mat(aie_hcwc8, "CDYX", "YCXD").reshape(out_c, out_h, out_w)
    raw_gold = np.loadtxt(fix + "golden_output.txt", delimiter=",", dtype=np.int8)
    gold_chw = raw_gold.reshape(out_c, out_h, out_w)

    diff = aie_chw.astype(np.int32) - gold_chw.astype(np.int32)
    n_match = int((diff == 0).sum())
    n_total = aie_chw.size
    max_d = int(np.abs(diff).max())
    mean_d = float(np.abs(diff).mean())
    pct = 100.0 * n_match / n_total
    print(
        f"PASS_CHAIN_E2E {args.mode}: {n_match}/{n_total} ({pct:.1f}%)  "
        f"max={max_d}  mean={mean_d:.3f}"
    )
    if max_d > args.atol:
        print(f"FAIL_CHAIN_E2E {args.mode}: max diff {max_d} exceeds atol={args.atol}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
