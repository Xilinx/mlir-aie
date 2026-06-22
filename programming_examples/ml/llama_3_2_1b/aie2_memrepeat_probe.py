"""Probe: validate the memtile write-once / read-Rx primitive for M3a.

A producer worker writes V fp32 values (in CHUNK chunks) into a memtile-resident
ObjectFifo; a consumer worker reads the whole V back R times via repeat_count
and copies each pass into a distinct slice of the output. The host then checks
out[pass*V : (pass+1)*V] == in for every pass. This isolates the exact IRON
spelling (forward/repeat_count/iter_count + 2-memtile split) the M3a
lm_head->sampler bridge needs, away from the GEMM/sampler kernels.

Env: LLAMA_PROBE_V (default 256), LLAMA_PROBE_CHUNK (default 64),
     LLAMA_PROBE_R (default 3), LLAMA_PROBE_SPLIT (1 = split across 2 memtiles).
"""

import argparse
import os as _os
import sys

import numpy as np

from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile, AnyMemTile
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern
from aie.dialects.arith import index_cast
from aie.ir import IntegerType

V = int(_os.environ.get("LLAMA_PROBE_V", "256"))
CHUNK = int(_os.environ.get("LLAMA_PROBE_CHUNK", "64"))
R = int(_os.environ.get("LLAMA_PROBE_R", "3"))
N_CHUNKS = V // CHUNK


def _f32(n):
    return np.ndarray[(int(n),), np.dtype[np.float32]]


def _idx(i):
    return index_cast(IntegerType.get_signless(32), i)


def factor(nb):
    if nb <= 1023:
        return (1, nb)
    for inner in range(min(nb, 1023), 0, -1):
        if nb % inner == 0 and nb // inner <= 1023:
            return (nb // inner, inner)
    raise ValueError(f"can't factor {nb}")


def chunk_tap(total_elems, base_elem, n_elems):
    outer, inner = factor(n_elems)
    return TensorAccessPattern(
        tensor_dims=[total_elems],
        offset=base_elem,
        sizes=[1, 1, outer, inner],
        strides=[0, 0, inner, 1],
    )


def build():
    rt_in_ty = _f32(V)
    rt_out_ty = _f32(V * R)
    HALF = V // 2
    t_half = _f32(HALF)

    KO = "llama_passthrough_f32.cc.o"
    k_copy = Kernel("passthrough_f32", KO, [t_half, t_half])

    # Split V across 2 memtiles: each holds a HALF resident, replayed R times.
    # This is the M3a logits geometry (fp32[V]=513KB > one 512KB memtile -> 2
    # halves of 256KB). Producer fills each half once; consumer reads each
    # half R times (R passes), copying to the out buffer interleaved per pass.
    of_in0 = ObjectFifo(t_half, depth=1, name="probe_in0")
    of_in1 = ObjectFifo(t_half, depth=1, name="probe_in1")
    # Pin each half to a DISTINCT memtile (row 1) so each 256 KB buffer gets
    # its own 512 KB memtile -- AnyMemTile piles both onto one tile and
    # overflows (513 KB > 512 KB).
    of_relay0 = of_in0.cons().forward(
        tile=Tile(0, 1), obj_type=t_half, depth=1, repeat_count=R, name="relay0"
    )
    of_relay1 = of_in1.cons().forward(
        tile=Tile(1, 1), obj_type=t_half, depth=1, repeat_count=R, name="relay1"
    )
    of_out = ObjectFifo(t_half, depth=2, name="probe_out")

    # Per pass: copy half0 then half1. Output layout per pass = [half0|half1] = V.
    def w_consume(c0, c1, c_out, k):
        for _ in range_(R):
            h0 = c0.acquire(1)
            o = c_out.acquire(1)
            k(h0, o)
            c_out.release(1)
            c0.release(1)
            h1 = c1.acquire(1)
            o = c_out.acquire(1)
            k(h1, o)
            c_out.release(1)
            c1.release(1)

    w = Worker(
        w_consume,
        [of_relay0.cons(), of_relay1.cons(), of_out.prod(), k_copy],
        tile=Tile(0, 2),
        stack_size=2048,
    )

    rt = Runtime()
    with rt.sequence(rt_in_ty, rt_out_ty) as (src, dst):
        rt.start(w)
        in_tg = rt.task_group()
        # src[0:HALF] -> in0, src[HALF:V] -> in1.
        rt.fill(of_in0.prod(), src, tap=chunk_tap(V, 0, HALF), task_group=in_tg)
        rt.fill(of_in1.prod(), src, tap=chunk_tap(V, HALF, HALF), task_group=in_tg)
        rt.finish_task_group(in_tg)
        out_tg = rt.task_group()
        rt.drain(of_out.cons(), dst, wait=True, task_group=out_tg)
        rt.finish_task_group(out_tg)

    return Program(NPU2(), rt).resolve_program()


def main():
    argparse.ArgumentParser().parse_args(sys.argv[1:])
    print(build())


if __name__ == "__main__":
    main()
