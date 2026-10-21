#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Validate mobilenet_numpy kernels against the brevitas per-bn fixtures
already checked into bottleneck_A/B/C/data/.

Usage:
    python3 test_numpy_per_bn.py
    # all 11 verified blocks should report 100.0% BIT-EXACT.

The fixtures use their OWN scales (bottleneck_A's scale_factors_per_bn.json
and each chain's scale_factors.json). This proves every kernel
arithmetic matches brevitas; mobilenet_numpy's docstring explains the full
network's gap to its golden (max 3, in post_l1's average pool).
"""

import dataclasses
import json
import os
import shutil
import sys
import tempfile

import numpy as np

from .mobilenet_numpy import _run_block
from .network_spec import block as nsblock

ROOT = os.path.dirname(__file__)
A_DATA = os.path.join(ROOT, "bottleneck_A/data/")
B_DATA = os.path.join(ROOT, "bottleneck_B/data/")
C_DATA = os.path.join(ROOT, "bottleneck_C/data/")


def _hwc(flat, c, h, w):
    """CHW int8 flat → HWC tensor."""
    return flat.reshape(c, h, w).transpose(1, 2, 0).copy()


def _check(label, out, gold):
    diff = out.astype(np.int32) - gold.astype(np.int32)
    n_match = int((diff == 0).sum())
    n_total = out.size
    max_d = int(np.abs(diff).max())
    mean_d = float(np.abs(diff).mean())
    pct = 100.0 * n_match / n_total
    verdict = "BIT-EXACT ✓" if max_d == 0 else f"max={max_d} mean={mean_d:.3f}"
    print(f"  {label:32s} {n_match:>6d}/{n_total:<6d} ({pct:5.1f}%)  {verdict}")
    return max_d == 0


def main():
    # ------------------------------------------------------------------
    # bottleneck_A: bn1, bn2, bn3, bn6, bn7, bn8 standalone
    # Each has its own input + golden + scales (each block in its own brevitas calibration).
    # ------------------------------------------------------------------
    print("=== bottleneck_A (per-bn standalone) ===")
    with open(A_DATA + "scale_factors_per_bn.json") as f:
        per_bn_sf = json.load(f)
    SHAPES = {
        "bn1": ((112, 112, 16), (56, 56, 24)),
        "bn2": ((56, 56, 24), (56, 56, 24)),
        "bn3": ((56, 56, 24), (28, 28, 40)),
        "bn6": ((28, 28, 40), (14, 14, 80)),
        "bn7": ((14, 14, 80), (14, 14, 80)),
        "bn8": ((14, 14, 80), (14, 14, 80)),
    }
    all_ok = True
    for name in ("bn1", "bn2", "bn3", "bn6", "bn7", "bn8"):
        in_shape, out_shape = SHAPES[name]
        H, W, IC = in_shape
        OH, OW, OC = out_shape
        # Per-bn block reads bnN_chain.txt (NOT bnN_single.txt — _run_block expects the chain naming).
        # We symlink in a temp dir... actually simpler: rename single -> chain at load.
        # Workaround: create a small staging dir with `bnN_chain.txt` -> single.
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(
                A_DATA + f"bn{name[2:]}_single.txt", tmp + f"/bn{name[2:]}_chain.txt"
            )
            inp = _hwc(
                np.loadtxt(
                    A_DATA + f"input_bn{name[2:]}_single.txt",
                    delimiter=",",
                    dtype=np.int8,
                ),
                IC,
                H,
                W,
            )
            out = _run_block(nsblock(name), inp, per_bn_sf, tmp + "/")
            gold = _hwc(
                np.loadtxt(
                    A_DATA + f"golden_output_bn{name[2:]}_single.txt",
                    delimiter=",",
                    dtype=np.int8,
                ),
                OC,
                OH,
                OW,
            )
            ok = _check(f"{name} (per-bn)", out, gold)
            all_ok &= ok

    # ------------------------------------------------------------------
    # bottleneck_A: bn0 → bn9 chain, against a golden generated without
    # bn5's residual (the device and the other checks keep it).
    # ------------------------------------------------------------------
    print("\n=== bottleneck_A (bn0 -> bn9 chain) ===")
    with open(A_DATA + "scale_factors_fused.json") as f:
        fused_sf = json.load(f)
    x = _hwc(
        np.loadtxt(A_DATA + "before_ifm_mem_fmt_1x1.txt", delimiter=",", dtype=np.int64)
        .astype(np.uint8),
        16,
        112,
        112,
    )
    for name in ("bn0", "bn1", "bn2", "bn3", "bn4", "bn5", "bn6", "bn7", "bn8", "bn9"):
        blk = nsblock(name)
        if name == "bn5":
            blk = dataclasses.replace(blk, skip=False)
        x = _run_block(blk, x, fused_sf, A_DATA)
    gold = _hwc(
        np.loadtxt(A_DATA + "golden_output.txt", delimiter=",", dtype=np.int64)
        .astype(np.uint8)
        .view(np.int8),
        80,
        14,
        14,
    )
    all_ok &= _check("bn0->bn9 chain, bn5 unskipped", x, gold)

    # ------------------------------------------------------------------
    # bottleneck_B: bn10 → bn11 → bn12 chain
    # ------------------------------------------------------------------
    print("\n=== bottleneck_B (bn10 -> bn11 -> bn12 chain) ===")
    B_SF = json.load(open(B_DATA + "scale_factors.json"))
    inp = _hwc(
        np.loadtxt(B_DATA + "before_ifm_mem_fmt_1x1.txt", delimiter=",", dtype=np.int8),
        80,
        14,
        14,
    )
    x = inp
    for name in ("bn10", "bn11", "bn12"):
        x = _run_block(nsblock(name), x, B_SF, B_DATA)
    gold = _hwc(
        np.loadtxt(B_DATA + "golden_output.txt", delimiter=",", dtype=np.int8), 80, 7, 7
    )
    all_ok &= _check("bn10->bn11->bn12 chain", x, gold)

    # ------------------------------------------------------------------
    # bottleneck_C: bn13 → bn14 cascade chain
    # ------------------------------------------------------------------
    print("\n=== bottleneck_C (bn13 -> bn14 cascade chain) ===")
    C_SF = json.load(open(C_DATA + "scale_factors.json"))
    inp = _hwc(
        np.loadtxt(C_DATA + "before_ifm_mem_fmt_1x1.txt", delimiter=",", dtype=np.int8),
        80,
        7,
        7,
    )
    x = inp
    for name in ("bn13", "bn14"):
        x = _run_block(nsblock(name), x, C_SF, C_DATA)
    gold = _hwc(
        np.loadtxt(C_DATA + "golden_output.txt", delimiter=",", dtype=np.int8), 80, 7, 7
    )
    all_ok &= _check("bn13->bn14 cascade chain", x, gold)

    print()
    if all_ok:
        print("ALL VERIFIED BLOCKS BIT-EXACT ✓")
        return 0
    else:
        print("FAIL — at least one block diverges")
        return 1


if __name__ == "__main__":
    sys.exit(main())
