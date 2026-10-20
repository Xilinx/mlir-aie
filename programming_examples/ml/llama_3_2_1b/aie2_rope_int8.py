#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Phase 2 RoPE (int8 in/out, bf16 cos/sin LUTs).

1 CT, 3 input fifos (x, cos, sin), 1 output. Single shared scale
(rope is norm-preserving). cos and sin come from precomputed LUTs;
in production they're per-position arrays.

CT DMA budget is 2 in + 2 out. 3 inputs here -- pack cos+sin into one
ObjectFifo to stay within budget.
"""

import argparse
import sys

import llama_kernels
import numpy as np
from aie.iron import CompileTime, ObjectFifo, Program, Runtime, TaskGroup, Worker, jit
from aie.iron.device import NPU2, Tile
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from ml_dtypes import bfloat16

ROPE_COL, ROPE_ROW = 4, 4

# Single per-tensor activation scale (same for in and out).
ACT_SCALE = 0.05


# DYN picks llama_rope_int8 (static scale, a scalar kernel arg) vs.
# llama_rope_int8_dyn (reads act_scale from x's own tail, passes it through
# unchanged for a downstream kernel).
@jit
def build(
    *,
    head_dim: CompileTime[int] = 64,
    n_heads: CompileTime[int] = 8,
    DYN: CompileTime[bool] = False,
):
    D = head_dim * n_heads
    cs_ty = np.ndarray[(2 * head_dim,), np.dtype[bfloat16]]  # cos || sin

    if DYN:
        x_ty = np.ndarray[(D + 8,), np.dtype[np.int8]]
        out_ty = np.ndarray[(D + 8,), np.dtype[np.int8]]
    else:
        x_ty = np.ndarray[(D,), np.dtype[np.int8]]
        out_ty = np.ndarray[(D,), np.dtype[np.int8]]

    of_x = ObjectFifo(x_ty, name="x")
    of_cs = ObjectFifo(cs_ty, name="cs")
    of_out = ObjectFifo(out_ty, name="out")

    if DYN:
        kernel = llama_kernels.rope(head_dim=head_dim, n_heads=n_heads).bind(
            "llama_rope_int8_dyn",
            [x_ty, cs_ty, out_ty],
        )

        def core_fn(c_x, c_cs, c_out, k):
            x = c_x.acquire(1)
            cs = c_cs.acquire(1)
            o = c_out.acquire(1)
            k(x, cs, o)
            c_x.release(1)
            c_cs.release(1)
            c_out.release(1)

    else:
        kernel = llama_kernels.rope(head_dim=head_dim, n_heads=n_heads).bind(
            "llama_rope_int8",
            [x_ty, cs_ty, out_ty, np.float32],
        )

        def core_fn(c_x, c_cs, c_out, k):
            x = c_x.acquire(1)
            cs = c_cs.acquire(1)
            o = c_out.acquire(1)
            k(x, cs, o, ACT_SCALE)
            c_x.release(1)
            c_cs.release(1)
            c_out.release(1)

    worker = Worker(
        core_fn,
        [of_x.cons(), of_cs.cons(), of_out.prod(), kernel],
        tile=Tile(ROPE_COL, ROPE_ROW),
    )

    def sequence(x, cs, o, x_prod, cs_prod, out_cons):
        tg = TaskGroup()
        x_prod.fill(x, group=tg)
        cs_prod.fill(cs, group=tg)
        out_cons.drain(o, wait=True, group=tg)
        tg.finish()

    rt = Runtime(
        sequence, [x_ty, cs_ty, out_ty, of_x.prod(), of_cs.prod(), of_out.cons()]
    )

    return Program(NPU2(), rt, workers=[worker]).resolve_program()


def main():
    p = argparse.ArgumentParser()
    add_compile_args(p, with_dev=True, default_dev="npu2", dev_choices=("npu2",))
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--dyn", action="store_true")
    opts = p.parse_args(sys.argv[1:])
    run_design_cli(
        build,
        opts,
        compile_kwargs={
            "head_dim": opts.head_dim,
            "n_heads": opts.n_heads,
            "DYN": opts.dyn,
        },
        device=NPU2(),
    )


if __name__ == "__main__":
    main()
