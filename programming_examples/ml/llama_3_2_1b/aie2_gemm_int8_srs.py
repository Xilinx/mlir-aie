"""Phase 1 dataflow frame for llama_gemm_int8_srs (single compute tile).

Wires up two input ObjectFifos (act, w_blob) and one output ObjectFifo
(out). The kernel itself is stubbed (`llama_gemm_int8_srs_pt`) -- it
ignores w_blob and copies act -> out. The purpose is to validate that
both inputs are delivered to the same compute tile and the output
round-trips back to DRAM.

In the real kernel, w_blob is a packed per-call payload (weights ||
bias || scale) delivered via StaticWeightStream, matching
cautious-eureka's aie2_llama_iron design. A compute tile only has 2
input + 2 output DMA channels, so packed payloads are the standard
way to bundle per-call constants.

Shapes are pinned tiny: M=8, K=64, N=64 (so M*N == M*K = 512 bytes and
passthrough is identity-shaped, trivially bit-exact-checkable).
"""

import argparse
import sys

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2


def build(M: int, K: int, N: int):
    # Real kernel: w_blob = N*K (weights, i8) + N*4 (bias, i32) + N*4 (scale, i32).
    # Stub ignores w_blob's contents -- we just need the bytes to flow.
    w_blob_bytes = N * K + N * 4 + N * 4

    act_ty    = np.ndarray[(M * K,),     np.dtype[np.int8]]
    w_blob_ty = np.ndarray[(w_blob_bytes,), np.dtype[np.int8]]
    out_ty    = np.ndarray[(M * N,),     np.dtype[np.int8]]

    of_act    = ObjectFifo(act_ty,    name="act")
    of_w_blob = ObjectFifo(w_blob_ty, name="w_blob")
    of_out    = ObjectFifo(out_ty,    name="out")

    kernel = Kernel(
        "llama_gemm_int8_srs_pt",
        "llama_gemm_int8_srs_pt.cc.o",
        [act_ty, w_blob_ty, out_ty],
    )

    def core_fn(of_act, of_w_blob, of_out, gemm):
        a = of_act.acquire(1)
        b = of_w_blob.acquire(1)
        o = of_out.acquire(1)
        gemm(a, b, o)
        of_act.release(1)
        of_w_blob.release(1)
        of_out.release(1)

    worker = Worker(
        core_fn,
        [of_act.cons(), of_w_blob.cons(), of_out.prod(), kernel],
    )

    rt = Runtime()
    with rt.sequence(act_ty, w_blob_ty, out_ty) as (a, w, o):
        rt.start(worker)
        rt.fill(of_act.prod(),    a)
        rt.fill(of_w_blob.prod(), w)
        rt.drain(of_out.cons(),   o, wait=True)

    return Program(NPU2(), rt).resolve_program()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-M", type=int, default=8)
    p.add_argument("-K", type=int, default=64)
    p.add_argument("-N", type=int, default=64)
    args = p.parse_args(sys.argv[1:])
    print(build(args.M, args.K, args.N))


if __name__ == "__main__":
    main()
