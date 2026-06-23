"""Probe: persistent on-device autoregressive decode loop (capstone structure).

A tiny tied LM (embed == lm_head, small V/D) runs T decode steps in ONE dispatch,
the token feeding back ON-CHIP every step (sampler -> next input), with NO host
involvement between tokens. Host: load the table once + seed token 0 + drain the
T generated tokens. This validates the full autoregressive feedback structure
that the real persistent loop needs (the real loop swaps the tiny step body for
the 16-layer chain + 262MB lm_head, weights host-streamed per token).

Mechanism: the worker holds the resident table (acquired once), loops T times:
  step 0: token from host seed; steps 1..T-1: token from a depth>=2 self-feedback
  fifo the worker itself produced. Each step: persist_decode_step(table, tok)->
  next_tok, written to BOTH the feedback fifo and the drained output.

Env: LLAMA_PD_V (vocab, 64), LLAMA_PD_D (dim, 64), LLAMA_PD_T (tokens, 8).
"""

import os as _os
import sys

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile
from aie.iron.controlflow import range_

V = int(_os.environ.get("LLAMA_PD_V", "64"))
D = int(_os.environ.get("LLAMA_PD_D", "64"))
T_STEPS = int(_os.environ.get("LLAMA_PD_T", "8"))


def _i8(n):
    return np.ndarray[(int(n),), np.dtype[np.int8]]


def _i32(n):
    return np.ndarray[(int(n),), np.dtype[np.int32]]


def build():
    t_table = _i8(V * D)
    t_tok = _i32(1)
    rt_table_ty = _i8(V * D)
    rt_seed_ty = _i32(1)
    rt_out_ty = _i32(T_STEPS)

    of_table = ObjectFifo(t_table, depth=1, name="pd_table")  # loaded once
    of_seed = ObjectFifo(t_tok, depth=1, name="pd_seed")  # token 0
    of_back = ObjectFifo(t_tok, depth=2, name="pd_back")  # self feedback (>=2)
    of_out = ObjectFifo(t_tok, depth=2, name="pd_out")  # drain each token

    KO = "llama_persist_decode.cc.o"
    k = Kernel("persist_decode_step", KO, [t_table, t_tok, t_tok])

    # Hold the table for the whole dispatch; loop T decode steps with the token
    # fed back on-chip (seed for step 0, of_back for steps 1..T-1).
    def w_decode(c_tbl, c_seed, c_back, c_fb, c_out, kern):
        tbl = c_tbl.acquire(1)  # resident table, held across all steps
        # step 0: seed token
        ti = c_seed.acquire(1)
        fb = c_fb.acquire(1)
        o = c_out.acquire(1)
        kern(tbl, ti, fb)  # fb = next token
        for i in range_(1):  # copy fb -> out (out is a separate drain buffer)
            o[0] = fb[0]
        c_seed.release(1)
        c_fb.release(1)
        c_out.release(1)
        # steps 1..T-1: token from feedback
        for _ in range_(T_STEPS - 1):
            ti = c_back.acquire(1)
            fb = c_fb.acquire(1)
            o = c_out.acquire(1)
            kern(tbl, ti, fb)
            for i in range_(1):
                o[0] = fb[0]
            c_back.release(1)
            c_fb.release(1)
            c_out.release(1)
        c_tbl.release(1)

    w = Worker(
        w_decode,
        [
            of_table.cons(),
            of_seed.cons(),
            of_back.cons(),
            of_back.prod(),
            of_out.prod(),
            k,
        ],
        tile=Tile(0, 2),
        stack_size=4096,
    )

    rt = Runtime()
    with rt.sequence(rt_table_ty, rt_seed_ty, rt_out_ty) as (table, seed, out):
        rt.start(w)
        s_tg = rt.task_group()
        rt.fill(of_table.prod(), table, task_group=s_tg)
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
