#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""End-to-end hardware test for one IRON block or chain design.

Compiles the design with @iron.jit, runs it on the NPU, and compares the output
against the matching brevitas golden fixture.

  block <bn>          per-block standalone (bn1, bn2, bn3, bn6, bn7, bn8), with
                      the per-bn fixtures and weights from bottleneck_A/data/.
  chain regular       bn0 -> bn9, fixtures from bottleneck_A/data/.
  chain pipeline      bn10 -> bn12, fixtures from bottleneck_B/data/.
  chain cascade       bn13 -> bn14 (with cascade weight DMAs), bottleneck_C/data/.

Usage (from programming_examples/ml):
    python3 -m mobilenet.test_e2e block bn3
    python3 -m mobilenet.test_e2e chain cascade --iters 20
"""

import argparse
import os
import sys

import aie.iron as iron
import numpy as np
from aie.utils.benchmark import print_benchmark, run_iters
from aie.utils.hostruntime import set_current_device
from aie.utils.hostruntime.argparse import (
    add_benchmark_args,
    add_compile_args,
    device_from_args,
)
from aie.utils.ml import DataShaper
from aie.utils.verify import Tolerance, compare

from .aie2_iron_chain import chain_design
from .aie2_iron_per_block import per_block_design

HERE = os.path.dirname(os.path.abspath(__file__))
VEC = 8

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

# Fixture directory and scale-factor file per chain; every block uses
# bottleneck_A with its per-bn scales and bnN_single.txt weights.
CHAINS = {
    "regular": ("bottleneck_A", "scale_factors_fused.json"),
    "pipeline": ("bottleneck_B", "scale_factors.json"),
    "cascade": ("bottleneck_C", "scale_factors.json"),
}


def _tolerance(key):
    if key == "chain:regular":
        # The placed-API bottleneck_A chain test accepted the same drift (#3009).
        return Tolerance.lsb(14, note="bottleneck_A chain drift, #3009")
    return Tolerance.exact()


def _loadtxt_i8(path):
    # bottleneck_A's IFM is in [0,255]; numpy 2.x rejects direct dtype=int8.
    return (
        np.loadtxt(path, delimiter=",", dtype=np.int64).astype(np.uint8).view(np.int8)
    )


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
        chunks.append(np.concatenate([put, get]))
    full = np.concatenate(chunks)
    assert full.size == 4 * 80 * 960
    return full


def _design(mode, target, fix):
    if mode == "block":
        kwargs = dict(
            block_name=target,
            data_dir=fix,
            scales_json=fix + "scale_factors_per_bn.json",
            wts_tag="single",
        )
        return per_block_design, kwargs
    kwargs = dict(mode=target, data_dir=fix, scales_json=fix + CHAINS[target][1])
    return chain_design, kwargs


def _make_argparser():
    p = argparse.ArgumentParser(description="Run one IRON mobilenet block or chain.")
    add_compile_args(p, default_dev="npu2")
    p.add_argument("mode", choices=["block", "chain"])
    p.add_argument(
        "target", help="block: bn1|bn2|bn3|bn6|bn7|bn8; chain: regular|pipeline|cascade"
    )
    # The NPU takes 6-13 launches after load to reach its steady latency.
    add_benchmark_args(p, default_warmup=20, default_iters=1)
    return p


def main():
    opts = _make_argparser().parse_args()
    key = f"{opts.mode}:{opts.target}"
    if key not in SHAPES:
        print(f"FAIL_E2E {key}: unsupported target")
        return 1
    set_current_device(device_from_args(opts, n_cols=None))
    in_w, in_h, in_c, out_w, out_h, out_c = SHAPES[key]
    bottleneck = "bottleneck_A" if opts.mode == "block" else CHAINS[opts.target][0]
    fix = os.path.join(HERE, bottleneck, "data") + "/"
    single = f"_bn{opts.target[2:]}_single" if opts.mode == "block" else ""
    ds = DataShaper()

    ifm = f"input{single}.txt" if single else "before_ifm_mem_fmt_1x1.txt"
    chw = _loadtxt_i8(fix + ifm).reshape(in_c, in_h, in_w)
    buffers = [
        iron.tensor(
            ds.reorder_mat(chw, "YCXC8", "CYX").flatten().view(np.int32),
            dtype=np.int32,
        )
    ]
    if key == "chain:cascade":
        buffers.append(
            iron.tensor(_load_cascade_weights(fix).view(np.int32), dtype=np.int32)
        )
    out = iron.zeros((out_w * out_h * out_c // 4,), dtype=np.int32)
    buffers.append(out)

    design, kwargs = _design(opts.mode, opts.target, fix)
    bench = run_iters(design, *buffers, warmup=opts.warmup, iters=opts.iters, **kwargs)

    # HCWC8 -> CHW, compared against the brevitas golden.
    actual = ds.reorder_mat(
        out.numpy().view(np.int8).reshape(out_h, out_c // VEC, out_w, VEC),
        "CDYX",
        "YCXD",
    ).reshape(out_c, out_h, out_w)
    golden = _loadtxt_i8(fix + f"golden_output{single}.txt")
    golden = golden.reshape(out_c, out_h, out_w)
    diff = np.abs(actual.astype(np.int32) - golden.astype(np.int32))
    verdict = compare(actual.astype(np.int32), golden.astype(np.int32), _tolerance(key))

    print_benchmark(bench)
    n_total = diff.size
    n_match = int((diff == 0).sum())
    status = "PASS_E2E" if verdict.ok else "FAIL_E2E"
    print(
        f"{status} {key} {n_match}/{n_total} ({100.0 * n_match / n_total:.1f}%)  "
        f"max={int(diff.max())}  mean={diff.mean():.3f}"
    )
    if not verdict.ok:
        print(verdict.detail)
    return 0 if verdict.ok else 1


if __name__ == "__main__":
    sys.exit(main())
