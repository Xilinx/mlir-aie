#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Phase 2 flowkv attention pair (qk + sv), v0 non-chunked full softmax.

Two CTs in a vertical column (col 0, rows 4 and 5): qk on CT0, sv on
CT1, with the bf16 probs vector flowing CT0 -> CT1 via a direct
neighbor ObjectFifo. Each CT runs within its 2-in/2-out DMA budget:

  CT0 (qk):  Q (shim) + K (shim) -> probs (CT->CT)
  CT1 (sv):  probs (CT->CT) + V (shim) -> out (shim)

3 runtime buffers (Q, K_packed||V_packed, out) -- but we keep K and V
as separate args here; the test combines/orders them appropriately.
Actually 4 runtime args (Q, K, V, out) -- DefaultNPURuntime tolerates
4 just fine; 5+ is where we segfault.

Scales baked at MLIR-gen time; the test uses the same values.
"""

import argparse
import sys

import llama_kernels
import numpy as np
from aie.iron import CompileTime, ObjectFifo, Program, Runtime, TaskGroup, Worker, jit
from aie.iron.device import NPU2, Tile
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli

PAIR_COL = 0
QK_ROW = 4
SV_ROW = 5

# Per-tensor scales baked at build time (test uses these).
Q_SCALE = 0.05
K_SCALE = 0.05
V_SCALE = 0.05
INV_OUT_SCALE = 1.0 / 0.05


# DYN picks llama_flowkv_{qk,sv} (static-scale scalar args) vs.
# llama_flowkv_{qk,sv}_dyn (scales read from q/k/v tails, same shape the
# real pipeline uses upstream of rope_int8_dyn/silu_mul_int8_dyn).
@jit
def build(
    *,
    head_dim: CompileTime[int] = 64,
    t: CompileTime[int] = 16,
    DYN: CompileTime[bool] = False,
):
    # fp32 probs on the CT0 -> CT1 stream; no bf16 truncation so the
    # softmax probabilities arrive bit-identical to what qk computed.
    probs_ty = np.ndarray[(t,), np.dtype[np.float32]]
    out_ty = np.ndarray[(head_dim,), np.dtype[np.int8]]

    if DYN:
        q_ty = np.ndarray[(head_dim + 8,), np.dtype[np.int8]]
        k_ty = np.ndarray[(8 + t * head_dim,), np.dtype[np.int8]]
        v_ty = np.ndarray[(8 + t * head_dim,), np.dtype[np.int8]]
    else:
        q_ty = np.ndarray[(head_dim,), np.dtype[np.int8]]
        k_ty = np.ndarray[(t * head_dim,), np.dtype[np.int8]]
        v_ty = np.ndarray[(t * head_dim,), np.dtype[np.int8]]

    of_q = ObjectFifo(q_ty, name="q")
    of_k = ObjectFifo(k_ty, name="k")
    of_v = ObjectFifo(v_ty, name="v")
    of_probs = ObjectFifo(probs_ty, name="probs")
    of_out = ObjectFifo(out_ty, name="out")

    unit = llama_kernels.flowkv(head_dim=head_dim, T=t)
    if DYN:
        k_qk = unit.bind(
            "llama_flowkv_qk_dyn",
            [q_ty, k_ty, probs_ty],
        )
        k_sv = unit.bind(
            "llama_flowkv_sv_dyn",
            [v_ty, probs_ty, out_ty],
        )

        def qk_fn(c_q, c_k, c_probs, k):
            q = c_q.acquire(1)
            kk = c_k.acquire(1)
            p = c_probs.acquire(1)
            k(q, kk, p)
            c_q.release(1)
            c_k.release(1)
            c_probs.release(1)

        def sv_fn(c_v, c_probs, c_out, k):
            v = c_v.acquire(1)
            p = c_probs.acquire(1)
            o = c_out.acquire(1)
            k(v, p, o)
            c_v.release(1)
            c_probs.release(1)
            c_out.release(1)

    else:
        k_qk = unit.bind(
            "llama_flowkv_qk",
            [q_ty, k_ty, probs_ty, np.float32, np.float32],
        )
        k_sv = unit.bind(
            "llama_flowkv_sv",
            [v_ty, probs_ty, out_ty, np.float32, np.float32],
        )

        def qk_fn(c_q, c_k, c_probs, k):
            q = c_q.acquire(1)
            kk = c_k.acquire(1)
            p = c_probs.acquire(1)
            k(q, kk, p, Q_SCALE, K_SCALE)
            c_q.release(1)
            c_k.release(1)
            c_probs.release(1)

        def sv_fn(c_v, c_probs, c_out, k):
            v = c_v.acquire(1)
            p = c_probs.acquire(1)
            o = c_out.acquire(1)
            k(v, p, o, V_SCALE, INV_OUT_SCALE)
            c_v.release(1)
            c_probs.release(1)
            c_out.release(1)

    worker_qk = Worker(
        qk_fn,
        [of_q.cons(), of_k.cons(), of_probs.prod(), k_qk],
        tile=Tile(PAIR_COL, QK_ROW),
    )
    worker_sv = Worker(
        sv_fn,
        [of_v.cons(), of_probs.cons(), of_out.prod(), k_sv],
        tile=Tile(PAIR_COL, SV_ROW),
    )

    def sequence(q, k, v, o, q_prod, k_prod, v_prod, out_cons):
        tg = TaskGroup()
        q_prod.fill(q, group=tg)
        k_prod.fill(k, group=tg)
        v_prod.fill(v, group=tg)
        out_cons.drain(o, wait=True, group=tg)
        tg.finish()

    rt = Runtime(
        sequence,
        [
            q_ty,
            k_ty,
            v_ty,
            out_ty,
            of_q.prod(),
            of_k.prod(),
            of_v.prod(),
            of_out.cons(),
        ],
    )

    return Program(NPU2(), rt, workers=[worker_qk, worker_sv]).resolve_program()


def main():
    p = argparse.ArgumentParser()
    add_compile_args(p, with_dev=True, default_dev="npu2", dev_choices=("npu2",))
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("-T", type=int, default=16)
    p.add_argument("--dyn", action="store_true")
    opts = p.parse_args(sys.argv[1:])
    run_design_cli(
        build,
        opts,
        compile_kwargs={"head_dim": opts.head_dim, "t": opts.T, "DYN": opts.dyn},
        device=NPU2(),
    )


if __name__ == "__main__":
    main()
