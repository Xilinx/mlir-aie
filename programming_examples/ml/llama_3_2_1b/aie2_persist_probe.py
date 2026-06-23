"""Probe: persistent on-device feedback loop (capstone primitive).

Proves the core mechanism the persistent autoregressive decode needs: a worker
seeded ONCE from the host iterates x = f(x) N times, feeding each iteration's
output back as the next iteration's input ON-CHIP (no host between iterations),
and drains each step's result for verification.

Mechanism (mirrors the chain's of_seed/of_back/w_router): iteration 0 consumes
the host seed; iterations 1..N-1 consume a self-feedback fifo the worker itself
produced. The loop trip count lives on the device. Host: 1 seed fill + 1 drain
of N results.

If this works, the real persistent loop wraps token->embed->layers->sample->
token in the same seed-once / feed-back-on-chip structure. Here f(x)=x+1 (a
trivial transform) to isolate the FEEDBACK plumbing from any compute.

Env: LLAMA_PP_N (iters, default 8), LLAMA_PP_LEN (vec len, default 64).
"""

import os as _os
import sys

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile, AnyMemTile
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern

N_ITER = int(_os.environ.get("LLAMA_PP_N", "8"))
LEN = int(_os.environ.get("LLAMA_PP_LEN", "64"))


def _i32(n):
    return np.ndarray[(int(n),), np.dtype[np.int32]]


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
        tensor_dims=[total],
        offset=base,
        sizes=[1, 1, outer, inner],
        strides=[0, 0, inner, 1],
    )


def build():
    t_vec = _i32(LEN)
    rt_seed_ty = _i32(LEN)
    rt_out_ty = _i32(LEN * N_ITER)

    of_seed = ObjectFifo(t_vec, depth=1, name="pp_seed")  # host seed (iter 0)
    of_back = ObjectFifo(t_vec, depth=2, name="pp_back")  # self feedback (>=2)
    of_out = ObjectFifo(t_vec, depth=2, name="pp_out")  # drain each iter

    KO = "llama_passthrough_f32.cc.o"
    k_step = Kernel("persist_step", KO, [t_vec, t_vec, t_vec])

    # Worker: iter 0 from seed, iters 1..N-1 from of_back (its own prior output).
    # Each iter produces BOTH the feedback (for next iter) and the drained out.
    def w_loop(c_seed, c_back, c_fb, c_out, k):
        x = c_seed.acquire(1)
        fb = c_fb.acquire(1)
        o = c_out.acquire(1)
        k(x, fb, o)  # fb = x+1 (next input), o = x+1 (drained)
        c_seed.release(1)
        c_fb.release(1)
        c_out.release(1)
        for _ in range_(N_ITER - 1):
            x = c_back.acquire(1)
            fb = c_fb.acquire(1)
            o = c_out.acquire(1)
            k(x, fb, o)
            c_back.release(1)
            c_fb.release(1)
            c_out.release(1)

    # of_back is produced AND consumed by the same worker -> self-feedback. Route
    # its buffer through a memtile (delegate) so prod/cons on one tile is legal.
    w = Worker(
        w_loop,
        [of_seed.cons(), of_back.cons(), of_back.prod(), of_out.prod(), k_step],
        tile=Tile(0, 2),
        stack_size=2048,
    )

    rt = Runtime()
    with rt.sequence(rt_seed_ty, rt_out_ty) as (seed, out):
        rt.start(w)
        s_tg = rt.task_group()
        rt.fill(of_seed.prod(), seed, task_group=s_tg)
        rt.finish_task_group(s_tg)
        o_tg = rt.task_group()
        rt.drain(of_out.cons(), out, wait=True, task_group=o_tg)
        rt.finish_task_group(o_tg)

    return Program(NPU2(), rt).resolve_program()


def main():
    import argparse

    argparse.ArgumentParser().parse_args(sys.argv[1:])
    print(build())


if __name__ == "__main__":
    main()
