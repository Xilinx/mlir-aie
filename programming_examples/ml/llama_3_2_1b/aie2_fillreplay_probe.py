"""Probe 3: validate the FULL logits relay (GEMM-fill + replay) in isolation.

A passthrough "GEMM" producer streams a HALF (chunk by chunk) into
LogitsHalfRelay; the relay holds it resident on a memtile and replays it R
times to a chunked consumer. Host checks every replayed chunk == the input.
This validates the new GEMM-fill side + the fill->replay handoff (the replay
side alone is already proven by aie2_replay_probe.py).

Env: LLAMA_FR_HALF (256), LLAMA_FR_CHUNK (64), LLAMA_FR_R (3).
"""

import argparse
import os as _os
import sys

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern

from logits_relay import LogitsHalfRelay

HALF = int(_os.environ.get("LLAMA_FR_HALF", "256"))
CHUNK = int(_os.environ.get("LLAMA_FR_CHUNK", "64"))
R = int(_os.environ.get("LLAMA_FR_R", "3"))
N_CHUNKS = HALF // CHUNK


def _f32(n):
    return np.ndarray[(int(n),), np.dtype[np.float32]]


def factor(nb):
    if nb <= 1023:
        return (1, nb)
    for inner in range(min(nb, 1023), 0, -1):
        if nb % inner == 0 and nb // inner <= 1023:
            return (nb // inner, inner)
    raise ValueError(f"can't factor {nb}")


def chunk_tap(total, base, n):
    outer, inner = factor(n)
    return TensorAccessPattern(
        tensor_dims=[total], offset=base, sizes=[1, 1, outer, inner],
        strides=[0, 0, inner, 1],
    )


def build():
    t_chunk = _f32(CHUNK)
    rt_in_ty = _f32(HALF)
    rt_out_ty = _f32(N_CHUNKS * R * CHUNK)

    of_src = ObjectFifo(t_chunk, depth=2, name="fr_src")
    of_out = ObjectFifo(t_chunk, depth=2, name="fr_out")

    relay = LogitsHalfRelay(
        half_elems=HALF,
        chunk_elems=CHUNK,
        repeat_count=R,
        memtile_placement=Tile(1, 1),
        gemm_placement=Tile(0, 3),
        sampler_placement=Tile(0, 2),
        name="fr",
        gemm_chunk_elems=CHUNK,
    )

    KO = "llama_passthrough_f32.cc.o"
    k_copy = Kernel("passthrough_f32_chunk", KO, [t_chunk, t_chunk])

    # "GEMM": pull each host chunk, write it into the relay's gemm send buffer.
    def w_gemm(c_src, rel, k):
        for _c in range_(N_CHUNKS):
            i = c_src.acquire(1)
            o = rel.gemm_acquire()
            k(i, o)
            rel.gemm_release()
            c_src.release(1)

    # Sampler: pull CHUNK windows, R replays.
    def w_samp(rel, c_out, k):
        for _ in range_(R):
            for _c in range_(N_CHUNKS):
                rb = rel.acquire(1)
                o = c_out.acquire(1)
                k(rb, o)
                c_out.release(1)
                rel.release(1)

    w_g = Worker(w_gemm, [of_src.cons(), relay, k_copy], tile=Tile(0, 3), stack_size=2048)
    w_s = Worker(w_samp, [relay, of_out.prod(), k_copy], tile=Tile(0, 2), stack_size=2048)

    rt = Runtime()
    with rt.sequence(rt_in_ty, rt_out_ty) as (src, dst):
        rt.start(w_g, w_s)
        in_tg = rt.task_group()
        rt.fill(of_src.prod(), src, task_group=in_tg)
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
