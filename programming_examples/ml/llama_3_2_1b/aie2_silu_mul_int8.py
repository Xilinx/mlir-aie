#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Phase 2 silu_mul (int8 in, int8 out, bf16 LUT-based SiLU).

1 CT, 2 input fifos (gate, up), 1 output. The SiLU LUT is baked into the kernel binary at
build time via gen_silu_lut.py + the silu_lut.h include, so gate_scale
is a build-time constant (not a kernel scalar arg). up_scale and
inv_out_scale remain runtime scalar args.
"""

import argparse
import os as _os
import sys

import llama_kernels
import numpy as np
from aie.iron import CompileTime, ObjectFifo, Program, Runtime, TaskGroup, Worker, jit
from aie.iron.device import NPU2, Tile
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli

SILU_COL, SILU_ROW = 4, 5

# Scales baked at build time: GATE_SCALE into the kernel's SiLU LUT, UP_SCALE
# and INV_OUT_SCALE here. Env-var overridable so the test can drive specific
# calibrations.
GATE_SCALE = float(_os.environ.get("SILU_GATE_SCALE", "0.05"))
UP_SCALE = float(_os.environ.get("SILU_UP_SCALE", "0.05"))
INV_OUT_SCALE = float(_os.environ.get("SILU_INV_OUT_SCALE", str(1.0 / 0.05)))


@jit
def build(*, D: CompileTime[int] = 8192, SELFCAL: CompileTime[bool] = False):
    i8_ty = np.ndarray[(D,), np.dtype[np.int8]]
    up_ty = np.ndarray[(D + 8,), np.dtype[np.int8]]  # up carries up_scale tail
    out_ty = np.ndarray[(D + 8,), np.dtype[np.int8]] if SELFCAL else i8_ty

    of_g = ObjectFifo(i8_ty, name="gate")
    of_u = ObjectFifo(up_ty, name="up")
    of_o = ObjectFifo(out_ty, name="out")

    if SELFCAL:
        kernel = llama_kernels.silu_mul(cols=D, gate_scale=GATE_SCALE).bind(
            "llama_silu_mul_int8_selfcal",
            [i8_ty, up_ty, out_ty],
        )

        def core_fn(c_g, c_u, c_o, k):
            g = c_g.acquire(1)
            u = c_u.acquire(1)
            o = c_o.acquire(1)
            k(g, u, o)
            c_g.release(1)
            c_u.release(1)
            c_o.release(1)

    else:
        kernel = llama_kernels.silu_mul(cols=D, gate_scale=GATE_SCALE).bind(
            "llama_silu_mul_int8_dyn",
            [i8_ty, up_ty, out_ty],
        )

        def core_fn(c_g, c_u, c_o, k):
            g = c_g.acquire(1)
            u = c_u.acquire(1)
            o = c_o.acquire(1)
            k(g, u, o)
            c_g.release(1)
            c_u.release(1)
            c_o.release(1)

    worker = Worker(
        core_fn,
        [of_g.cons(), of_u.cons(), of_o.prod(), kernel],
        tile=Tile(SILU_COL, SILU_ROW),
    )

    def sequence(g, u, o, g_prod, u_prod, o_cons):
        tg = TaskGroup()
        g_prod.fill(g, group=tg)
        u_prod.fill(u, group=tg)
        o_cons.drain(o, wait=True, group=tg)
        tg.finish()

    rt = Runtime(
        sequence, [i8_ty, up_ty, out_ty, of_g.prod(), of_u.prod(), of_o.cons()]
    )

    return Program(NPU2(), rt, workers=[worker]).resolve_program()


def main():
    p = argparse.ArgumentParser()
    add_compile_args(p, with_dev=True, default_dev="npu2", dev_choices=("npu2",))
    p.add_argument("-D", type=int, default=8192)
    p.add_argument("--selfcal", action="store_true")
    opts = p.parse_args(sys.argv[1:])
    run_design_cli(
        build,
        opts,
        compile_kwargs={"D": opts.D, "SELFCAL": opts.selfcal},
        device=NPU2(),
    )


if __name__ == "__main__":
    main()
