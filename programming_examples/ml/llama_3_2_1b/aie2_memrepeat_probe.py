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
    CHUNK_PER_HALF = HALF // CHUNK  # chunks the producer emits per half
    t_half = _f32(HALF)
    t_chunk = _f32(CHUNK)

    KO = "llama_passthrough_f32.cc.o"
    k_copy = Kernel("passthrough_f32_chunk", KO, [t_chunk, t_chunk])

    # GEMM-side geometry test: a COMPUTE-tile producer emits CHUNK-sized buffers
    # (like the lm_head GEMM's per-tile output); a memtile relay ASSEMBLES the
    # chunks into a full HALF buffer (obj_type=t_half) and replays it R times.
    # This is the real M3a producer pattern (chunked producer, whole-buffer
    # replay), which the earlier host-fill probe did not exercise.
    of_src0 = ObjectFifo(t_chunk, depth=2, name="src0")  # host -> producer tile
    of_src1 = ObjectFifo(t_chunk, depth=2, name="src1")
    # Producer (compute tile) emits CHUNK buffers into of_prod; .forward() to a
    # memtile relay whose CONSUMER side has depth=CHUNK_PER_HALF -> the memtile
    # holds the WHOLE half (all chunks) resident; repeat_count=R replays the
    # full multi-buffer set R times (the lowering's multi-buffer-repeat path,
    # NOT delegate_tile which conflicts with repeat_count). Both endpoints are
    # CHUNK-typed so neither compute tile holds a 256KB buffer.
    of_prod0 = ObjectFifo(t_chunk, depth=2, name="prod0")
    of_prod1 = ObjectFifo(t_chunk, depth=2, name="prod1")
    of_relay0 = of_prod0.cons().forward(
        tile=Tile(0, 1),
        obj_type=t_chunk,
        depth=CHUNK_PER_HALF,
        repeat_count=R,
        name="relay0",
    )
    of_relay1 = of_prod1.cons().forward(
        tile=Tile(1, 1),
        obj_type=t_chunk,
        depth=CHUNK_PER_HALF,
        repeat_count=R,
        name="relay1",
    )

    of_out = ObjectFifo(t_chunk, depth=2, name="probe_out")

    # Producer: copy each host chunk into the memtile relay (chunk-typed).
    def w_produce(c_in, c_out, k):
        for _ch in range_(CHUNK_PER_HALF):
            i = c_in.acquire(1)
            o = c_out.acquire(1)
            k(i, o)
            c_in.release(1)
            c_out.release(1)

    # Consumer: read CHUNK pieces, R replays x CHUNK_PER_HALF chunks per half.
    def w_consume(c0, c1, c_out, k):
        for _ in range_(R):
            for _ch in range_(CHUNK_PER_HALF):
                h = c0.acquire(1)
                o = c_out.acquire(1)
                k(h, o)
                c_out.release(1)
                c0.release(1)
            for _ch in range_(CHUNK_PER_HALF):
                h = c1.acquire(1)
                o = c_out.acquire(1)
                k(h, o)
                c_out.release(1)
                c1.release(1)

    w_prod0 = Worker(
        w_produce,
        [of_src0.cons(), of_prod0.prod(), k_copy],
        tile=Tile(0, 3),
        stack_size=2048,
    )
    w_prod1 = Worker(
        w_produce,
        [of_src1.cons(), of_prod1.prod(), k_copy],
        tile=Tile(1, 3),
        stack_size=2048,
    )
    w_cons = Worker(
        w_consume,
        [of_relay0.cons(), of_relay1.cons(), of_out.prod(), k_copy],
        tile=Tile(0, 2),
        stack_size=2048,
    )

    rt = Runtime()
    with rt.sequence(rt_in_ty, rt_out_ty) as (src, dst):
        rt.start(w_prod0, w_prod1, w_cons)
        in_tg = rt.task_group()
        rt.fill(of_src0.prod(), src, tap=chunk_tap(V, 0, HALF), task_group=in_tg)
        rt.fill(of_src1.prod(), src, tap=chunk_tap(V, HALF, HALF), task_group=in_tg)
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
