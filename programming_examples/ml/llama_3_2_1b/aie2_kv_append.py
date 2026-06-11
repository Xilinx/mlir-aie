"""Stage 1 unit: isolated llama_kv_append_head kernel.

Validates the on-chip KV append ARITHMETIC (rope_k + per-slot dynamic quant +
slot write) bit-exact vs the numpy `position=` oracle, BEFORE the full
single-layer dataflow integration. One worker, one KV head: in = k_fp, v_fp
(fp32[HEAD_DIM]), cs (bf16[2*HEAD_DIM]), and the per-head cache (host-filled,
position in its prefix); out = the updated cache (drained back).
"""

import argparse
import sys

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile

HEAD_D = 64
T = 128
PREFIX = 8
SCALE_BYTES = T * 4
BODY_BYTES = T * HEAD_D
PER_HEAD = PREFIX + 2 * (SCALE_BYTES + BODY_BYTES)


def _i8(n):
    return np.ndarray[(int(n),), np.dtype[np.int8]]


def _f32(n):
    return np.ndarray[(int(n),), np.dtype[np.float32]]


def _bf16(n):
    from ml_dtypes import bfloat16

    return np.ndarray[(int(n),), np.dtype[bfloat16]]


KVFP_BYTES = HEAD_D * 4 + HEAD_D * 4 + 2 * HEAD_D * 2  # k_fp | v_fp | cs(bf16)


def build():
    t_kvfp = _i8(KVFP_BYTES)
    t_kv = _i8(PER_HEAD)

    KO = "llama_kv_append.cc.o"
    k_app = Kernel("llama_kv_append_head", KO, [t_kvfp, t_kv, t_kv])

    def w_app(c_fp, c_in, c_out, k):
        fp = c_fp.acquire(1)
        kin = c_in.acquire(1)
        kout = c_out.acquire(1)
        k(fp, kin, kout)
        c_fp.release(1)
        c_in.release(1)
        c_out.release(1)

    of_fp = ObjectFifo(t_kvfp, depth=1, name="kvfp")
    of_kv_in = ObjectFifo(t_kv, depth=1, name="kv_in")
    of_kv_out = ObjectFifo(t_kv, depth=1, name="kv_out")

    worker = Worker(
        w_app,
        [of_fp.cons(), of_kv_in.cons(), of_kv_out.prod(), k_app],
        tile=Tile(0, 2),
        stack_size=16384,
    )

    rt = Runtime()
    with rt.sequence(t_kvfp, t_kv, t_kv) as (kvfp, kv_in, kv_out):
        rt.start(worker)
        rt.fill(of_fp.prod(), kvfp)
        rt.fill(of_kv_in.prod(), kv_in)
        rt.drain(of_kv_out.cons(), kv_out, wait=True)

    return Program(NPU2(), rt).resolve_program()


def main():
    argparse.ArgumentParser().parse_args(sys.argv[1:])
    print(build())


if __name__ == "__main__":
    main()
