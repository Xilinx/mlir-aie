#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Standalone block-wise attention: one KV head's live cache split across cores.

The host KV cache is block-major int32 [block][layer][half][head][TB rows],
a row being [k int8[64] | v int8[64] | k_scale f32 | v_scale f32]. Core c
streams its contiguous block range of one (layer, half, head) and folds each
block into its state; the memtile joins the cores' states and a merge core
folds them in core order and requantizes.
"""

import argparse
import sys

import llama_kernels
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, ObjectFifo, Program, Runtime, TaskGroup, Worker, jit
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from numpy_attn_block import core_ranges

HEAD_DIM = 64
REP = 4
TB = 64
ROW_BYTES = 2 * HEAD_DIM + 8
BLOCK_WORDS = TB * ROW_BYTES // 4
STATE_BYTES = 64 + REP * (64 + 4 * HEAD_DIM)
Q_BYTES = REP * HEAD_DIM + REP * 8
OUT_BYTES = REP * HEAD_DIM + REP * 4


def kv_words(t, n_layers, n_heads):
    """int32 words of the block-major cache holding `t` rows.

    Args:
        `t`: live rows.
        `n_layers`: layers in the cache.
        `n_heads`: KV heads per half.

    Returns:
        The cache size in int32 words.
    """
    return -(-t // TB) * n_layers * 2 * n_heads * BLOCK_WORDS


@jit
def build(
    *,
    T: CompileTime[int] = 64,
    C: CompileTime[int] = 1,
    N_LAYERS: CompileTime[int] = 2,
    N_HEADS: CompileTime[int] = 4,
    LAYER: CompileTime[int] = 1,
    HALF: CompileTime[int] = 1,
    HEAD: CompileTime[int] = 2,
):
    q_ty = np.ndarray[(Q_BYTES,), np.dtype[np.int8]]
    blk_ty = np.ndarray[(BLOCK_WORDS,), np.dtype[np.int32]]
    state_ty = np.ndarray[(STATE_BYTES,), np.dtype[np.int8]]
    parts_ty = np.ndarray[(C * STATE_BYTES,), np.dtype[np.int8]]
    out_ty = np.ndarray[(OUT_BYTES,), np.dtype[np.int8]]
    total = kv_words(T, N_LAYERS, N_HEADS)
    kv_ty = np.ndarray[(total,), np.dtype[np.int32]]

    unit = llama_kernels.attn_block()
    k_init = unit.bind("llama_attn_state_init", [state_ty])
    k_block = unit.bind("llama_attn_block", [q_ty, blk_ty, state_ty, np.int32])
    k_merge = unit.bind("llama_attn_merge_selfcal", [parts_ty, out_ty, np.int32])

    n_blocks = -(-T // TB)
    ranges = core_ranges(n_blocks, C)
    last_rows = T - (n_blocks - 1) * TB

    of_q = ObjectFifo(q_ty, depth=1, name="q")
    of_parts = ObjectFifo(parts_ty, depth=1, name="parts")
    part_fifos = of_parts.prod().join(
        [c * STATE_BYTES for c in range(C)],
        obj_types=[state_ty] * C,
        depths=[1] * C,
        names=[f"part{c}" for c in range(C)],
    )
    of_out = ObjectFifo(out_ty, depth=1, name="out")
    kv_fifos = [
        ObjectFifo(blk_ty, depth=2, name=f"kv{c}") if end > first else None
        for c, (first, end) in enumerate(ranges)
    ]

    workers = []
    for c, (first, end) in enumerate(ranges):
        n_full = end - first - (1 if end == n_blocks else 0)
        tail_rows = last_rows if end == n_blocks else 0

        def attend(q_c, part_p, init, block, *kv, n_full=n_full, tail_rows=tail_rows):
            q = q_c.acquire(1)
            st = part_p.acquire(1)
            init(st)
            if kv:
                (kv_c,) = kv
                if n_full:
                    for _ in range_(n_full):
                        kb = kv_c.acquire(1)
                        block(q, kb, st, TB)
                        kv_c.release(1)
                if tail_rows:
                    kb = kv_c.acquire(1)
                    block(q, kb, st, tail_rows)
                    kv_c.release(1)
            part_p.release(1)
            q_c.release(1)

        args = [of_q.cons(), part_fifos[c].prod(), k_init, k_block]
        kv_fifo = kv_fifos[c]
        if kv_fifo is not None:
            args.append(kv_fifo.cons())
        workers.append(Worker(attend, args, while_true=False))

    def merge(parts_c, out_p, k):
        parts = parts_c.acquire(1)
        o = out_p.acquire(1)
        k(parts, o, C)
        parts_c.release(1)
        out_p.release(1)

    workers.append(
        Worker(merge, [of_parts.cons(), of_out.prod(), k_merge], while_true=False)
    )

    block_stride = N_LAYERS * 2 * N_HEADS * BLOCK_WORDS
    head_off = ((LAYER * 2 + HALF) * N_HEADS + HEAD) * BLOCK_WORDS
    kv_prods = [f.prod() for f in kv_fifos if f is not None]
    kv_ranges = [r for r, f in zip(ranges, kv_fifos) if f is not None]

    def sequence(q, kv, o, q_prod, out_cons, *kv_ps):
        tg = TaskGroup()
        q_prod.fill(q, group=tg)
        for kv_p, (first, end) in zip(kv_ps, kv_ranges):
            kv_p.fill(
                kv,
                tap=TensorAccessPattern(
                    tensor_dims=[total],
                    offset=first * block_stride + head_off,
                    sizes=[1, end - first, 4, BLOCK_WORDS // 4],
                    strides=[0, block_stride, BLOCK_WORDS // 4, 1],
                ),
                group=tg,
            )
        out_cons.drain(o, wait=True, group=tg)
        tg.finish()

    rt = Runtime(
        sequence,
        [q_ty, kv_ty, out_ty, of_q.prod(), of_out.cons(), *kv_prods],
    )
    return Program(NPU2(), rt, workers=workers).resolve_program()


def main():
    p = argparse.ArgumentParser()
    add_compile_args(p, with_dev=True, default_dev="npu2", dev_choices=("npu2",))
    p.add_argument("-T", type=int, default=64)
    p.add_argument("-C", type=int, default=1, choices=(1, 2, 4))
    opts = p.parse_args(sys.argv[1:])
    run_design_cli(
        build, opts, compile_kwargs={"T": opts.T, "C": opts.C}, device=NPU2()
    )


if __name__ == "__main__":
    main()
