#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
"""End-to-end hardware test for one IRON block or chain design.

Two modes:

  block <bn>          per-block standalone (bn1, bn2, bn3, bn6, bn7, bn8).
                      Inputs: input_bnN_single.txt + golden_output_bnN_single.txt
                      from bottleneck_A/data/.

  chain pipeline      bn10 -> bn11 -> bn12. Inputs: before_ifm_mem_fmt_1x1.txt +
                      golden_output.txt from bottleneck_B/data/.
  chain cascade       bn13 -> bn14 (with cascade weight DMAs). Same fixture
                      pattern, fixtures from bottleneck_C/data/.

Compares the NPU output bit-exact against the brevitas golden reference.

Usage:
    python3 test_e2e.py block bn3 --xclbin <p> --insts <p> \\
        --fixture-dir bottleneck_A/data
    python3 test_e2e.py chain pipeline --xclbin <p> --insts <p> \\
        --fixture-dir bottleneck_B/data
"""

import argparse
import sys
import numpy as np

import aie.iron as iron
from aie.utils.ml import DataShaper
from aie.utils import DefaultNPURuntime, NPUKernel

# (in_w, in_h, in_c, out_w, out_h, out_c) per supported test target.
SHAPES = {
    "block:bn1": (112, 112, 16, 56, 56, 24),
    "block:bn2": (56, 56, 24, 56, 56, 24),
    "block:bn3": (56, 56, 24, 28, 28, 40),
    "block:bn6": (28, 28, 40, 14, 14, 80),
    "block:bn7": (14, 14, 80, 14, 14, 80),
    "block:bn8": (14, 14, 80, 14, 14, 80),
    "chain:regular": (112, 112, 16, 14, 14, 80),
    "chain:pipeline": (14, 14, 80, 7, 7, 80),
    "chain:cascade": (7, 7, 80, 7, 7, 80),
}
VEC = 8


def _load_input_chw(fix, mode, target, shape):
    """Return the (CHW) int8 input tensor for this test target."""
    in_c, in_h, in_w = shape
    if mode == "block":
        path = fix + f"input_bn{target[2:]}_single.txt"
    else:
        path = fix + "before_ifm_mem_fmt_1x1.txt"
    # bottleneck_A's IFM is in [0,255]; numpy 2.x rejects direct dtype=int8.
    raw = np.loadtxt(path, delimiter=",", dtype=np.int64).astype(np.uint8).view(np.int8)
    assert raw.size == in_h * in_w * in_c, f"{path}: {raw.size} != {in_h*in_w*in_c}"
    return raw.reshape(in_c, in_h, in_w)


def _load_golden_chw(fix, mode, target, shape):
    """Return the (CHW) int8 golden tensor for this test target."""
    out_c, out_h, out_w = shape
    if mode == "block":
        path = fix + f"golden_output_bn{target[2:]}_single.txt"
    else:
        path = fix + "golden_output.txt"
    raw = np.loadtxt(path, delimiter=",", dtype=np.int64).astype(np.uint8).view(np.int8)
    assert (
        raw.size == out_h * out_w * out_c
    ), f"{path}: {raw.size} != {out_h*out_w*out_c}"
    return raw.reshape(out_c, out_h, out_w)


def _load_cascade_weights(fix):
    """Concatenate the 4 cascade weight chunks (bn13_l1 | bn13_l3 | bn14_l1 | bn14_l3).

    Mirrors aie2_iron_chain.py's cascade rt.sequence: ONE host buffer, sliced
    by TensorAccessPatterns inside the runtime. Each chunk is 80*960=76800 B.
    """
    chunks = []
    for bn in ("bn13", "bn14"):
        chunks.append(
            np.loadtxt(fix + f"{bn}_1_chain.txt", delimiter=",", dtype=np.int8)
        )
        put = np.loadtxt(fix + f"{bn}_3_put_chain.txt", delimiter=",", dtype=np.int8)
        get = np.loadtxt(fix + f"{bn}_3_get_chain.txt", delimiter=",", dtype=np.int8)
        chunks.append(np.concatenate([put, get]))  # bn_l3 (put + get)
    full = np.concatenate(chunks)
    assert full.size == 4 * 80 * 960
    return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["block", "chain"])
    ap.add_argument(
        "target", help="block: bn1|bn2|bn3|bn6|bn7|bn8; chain: pipeline|cascade"
    )
    ap.add_argument("--xclbin", required=True)
    ap.add_argument("--insts", required=True)
    ap.add_argument("--fixture-dir", required=True)
    ap.add_argument("--atol", type=int, default=0)
    args = ap.parse_args()

    key = f"{args.mode}:{args.target}"
    if key not in SHAPES:
        print(f"FAIL_E2E {key}: unsupported target")
        return 1
    in_w, in_h, in_c, out_w, out_h, out_c = SHAPES[key]
    fix = args.fixture_dir.rstrip("/") + "/"

    npu = NPUKernel(args.xclbin, args.insts)
    handle = DefaultNPURuntime.load(npu)
    ds = DataShaper()

    # Input: CHW int8 → YCXC8 layout the AIE design expects.
    chw = _load_input_chw(fix, args.mode, args.target, (in_c, in_h, in_w))
    in_tensor = iron.tensor(
        ds.reorder_mat(chw, "YCXC8", "CYX").flatten().view(np.int32), dtype=np.int32
    )
    out_tensor = iron.zeros((out_w * out_h * out_c // 4,), dtype=np.int32)

    buffers = [in_tensor]
    if key == "chain:cascade":
        buffers.append(
            iron.tensor(_load_cascade_weights(fix).view(np.int32), dtype=np.int32)
        )
    buffers.append(out_tensor)

    print(f"  Running {args.mode} {args.target} on NPU ...")
    DefaultNPURuntime.run(handle, buffers)

    # Decode output: HCWC8 → CHW; bit-exact compare vs brevitas golden.
    aie_chw = ds.reorder_mat(
        out_tensor.numpy().view(np.int8).reshape(out_h, out_c // VEC, out_w, VEC),
        "CDYX",
        "YCXD",
    ).reshape(out_c, out_h, out_w)
    gold = _load_golden_chw(fix, args.mode, args.target, (out_c, out_h, out_w))

    diff = aie_chw.astype(np.int32) - gold.astype(np.int32)
    n_match = int((diff == 0).sum())
    n_total = aie_chw.size
    max_d = int(np.abs(diff).max())
    mean_d = float(np.abs(diff).mean())
    pct = 100.0 * n_match / n_total
    print(
        f"PASS_E2E {args.mode}:{args.target} {n_match}/{n_total} ({pct:.1f}%)  "
        f"max={max_d}  mean={mean_d:.3f}"
    )
    if max_d > args.atol:
        print(
            f"FAIL_E2E {args.mode}:{args.target} max diff {max_d} exceeds atol={args.atol}"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
