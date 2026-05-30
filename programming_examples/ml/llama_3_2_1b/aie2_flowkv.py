"""Phase 2 flowkv attention pair (qk + sv), v0 non-chunked full softmax.

Two CTs in a vertical column (col 0, rows 4 and 5, matching
DECODE_PLACEMENT["attention"]["pair0"]). qk on CT0, sv on CT1, with
the bf16 probs vector flowing CT0 -> CT1 via a direct neighbor
ObjectFifo. Each CT runs within its 2-in/2-out DMA budget:

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

import numpy as np
from ml_dtypes import bfloat16

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile


PAIR_COL = 0
QK_ROW = 4
SV_ROW = 5

# Per-tensor scales baked at build time (test uses these).
Q_SCALE       = 0.05
K_SCALE       = 0.05
V_SCALE       = 0.05
INV_OUT_SCALE = 1.0 / 0.05


def build(head_dim: int, t: int):
    q_ty     = np.ndarray[(head_dim,),       np.dtype[np.int8]]
    kv_ty    = np.ndarray[(t * head_dim,),   np.dtype[np.int8]]
    probs_ty = np.ndarray[(t,),              np.dtype[bfloat16]]
    out_ty   = np.ndarray[(head_dim,),       np.dtype[np.int8]]

    of_q     = ObjectFifo(q_ty,     name="q")
    of_k     = ObjectFifo(kv_ty,    name="k")
    of_v     = ObjectFifo(kv_ty,    name="v")
    of_probs = ObjectFifo(probs_ty, name="probs")
    of_out   = ObjectFifo(out_ty,   name="out")

    k_qk = Kernel(
        "llama_flowkv_qk",
        "llama_flowkv.cc.o",
        [q_ty, kv_ty, probs_ty, np.float32, np.float32],
    )
    k_sv = Kernel(
        "llama_flowkv_sv",
        "llama_flowkv.cc.o",
        [kv_ty, probs_ty, out_ty, np.float32, np.float32],
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

    rt = Runtime()
    with rt.sequence(q_ty, kv_ty, kv_ty, out_ty) as (q, k, v, o):
        rt.start(worker_qk, worker_sv)
        rt.fill(of_q.prod(), q)
        rt.fill(of_k.prod(), k)
        rt.fill(of_v.prod(), v)
        rt.drain(of_out.cons(), o, wait=True)

    return Program(NPU2(), rt).resolve_program()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("-T", type=int, default=16)
    args = p.parse_args(sys.argv[1:])
    print(build(args.head_dim, args.T))


if __name__ == "__main__":
    main()
