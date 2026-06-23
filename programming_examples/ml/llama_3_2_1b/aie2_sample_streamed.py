"""Standalone streamed sampler over a DDR fp32[V] logits buffer (M2 of the
on-chip sample tile). Real vocab V=128256.

The full sampler (temperature + top-k + exp-LUT softmax + xoshiro inverse-CDF
draw, greedy as the temp<=0 short-circuit) needs several ORDERED passes over
all V logits, which do not fit L1. So the host streams the same DDR logits
buffer through an L1 chunk fifo THREE times (one fill loop per pass); a single
sampler tile carries scalar/heap state in a held-once L1 scratch buffer across
all chunks and passes, then a finalize call writes the int32 token.

Runtime sequence:
   logits  (fp32[V])      # precomputed lm_head logits (host numpy for M2;
                          #   the chain GEMM in M3)
   params  (uint32[3])    # [temperature bits, top_k, seed]
   scratch (int8[STATE])  # held-once sampler carry state (zeroed by host)
   token   (int32[1])     # output token id
"""

import argparse
import os as _os
import sys

import numpy as np

from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern
from aie.dialects.arith import index_cast
from aie.ir import IntegerType

VOCAB = int(_os.environ.get("LLAMA_SAMPLE_V", "128256"))
CHUNK_N = int(_os.environ.get("LLAMA_SAMPLE_CHUNK", "2004"))  # V=64*2004
N_CHUNKS = VOCAB // CHUNK_N  # 64
N_PASSES = 3
STATE_BYTES = 1088  # 64 B header + 256 fp32 top-k set (OFF_HEAP=64)

SAMPLE_COL, SAMPLE_ROW = 5, 5


def _i8(n):
    return np.ndarray[(int(n),), np.dtype[np.int8]]


def _f32(n):
    return np.ndarray[(int(n),), np.dtype[np.float32]]


def _u32(n):
    return np.ndarray[(int(n),), np.dtype[np.uint32]]


def _i32(n):
    return np.ndarray[(int(n),), np.dtype[np.int32]]


def _idx(i):
    return index_cast(IntegerType.get_signless(32), i)


def _const_i32(v):
    from aie.dialects.arith import constant

    return constant(IntegerType.get_signless(32), v)


def factor(nb):
    if nb <= 1023:
        return (1, nb)
    for inner in range(min(nb, 1023), 0, -1):
        if nb % inner == 0 and nb // inner <= 1023:
            return (nb // inner, inner)
    raise ValueError(f"can't factor nbytes={nb} into two <=1023 dims")


def chunk_tap(total_elems, base_elem, n_elems):
    """One contiguous chunk of n_elems fp32 from a fp32[total_elems] tensor.
    TAP dims are in ELEMENTS (the tensor is fp32-typed), not bytes."""
    outer, inner = factor(n_elems)
    return TensorAccessPattern(
        tensor_dims=[total_elems],
        offset=base_elem,
        sizes=[1, 1, outer, inner],
        strides=[0, 0, inner, 1],
    )


def build():
    rt_logits_ty = _f32(VOCAB)
    rt_params_ty = _u32(3)
    rt_token_ty = _i32(1)

    t_chunk = _f32(CHUNK_N)
    t_state = _i8(STATE_BYTES)
    t_params = _u32(3)
    t_token = _i32(1)

    of_logits = ObjectFifo(t_chunk, depth=2, name="logits")
    of_params = ObjectFifo(t_params, depth=1, name="params")
    of_token = ObjectFifo(t_token, name="token")
    # Sampler carry state: worker-local L1 Buffer (NOT a DMA fifo). Persists
    # across all chunk calls + the 3 passes within one dispatch. Keeps the
    # tile within the 2-in/2-out DMA budget (logits+params in, token out).
    b_state = Buffer(t_state, name="sample_state")

    KO = "llama_sample_streamed.cc.o"
    k_sample = Kernel(
        "llama_sample_streamed",
        KO,
        [t_chunk, t_state, t_params, np.int32, np.int32, np.int32],
    )
    k_final = Kernel(
        "llama_sample_streamed_finalize", KO, [t_state, t_token, t_params]
    )

    # Hold params for the whole dispatch; stream logit chunks 3x. State is the
    # worker-local Buffer (st), self-seeded by the kernel at chunk 0 of pass 1.
    # local_chunk=0: the fifo IS chunk-typed, so each acquire is one chunk.
    def w_sample(c_log, st, c_params, c_tok, ks, kf):
        p = c_params.acquire(1)
        tok = c_tok.acquire(1)
        for pass_i in range_(1, N_PASSES + 1):
            for ch in range_(N_CHUNKS):
                lg = c_log.acquire(1)
                ks(lg, st, p, _idx(pass_i), _idx(ch), _const_i32(0))
                c_log.release(1)
        kf(st, tok, p)
        c_params.release(1)
        c_tok.release(1)

    worker = Worker(
        w_sample,
        [
            of_logits.cons(),
            b_state,
            of_params.cons(),
            of_token.prod(),
            k_sample,
            k_final,
        ],
        tile=Tile(SAMPLE_COL, SAMPLE_ROW),
        stack_size=8192,
    )

    rt = Runtime()
    with rt.sequence(rt_logits_ty, rt_params_ty, rt_token_ty) as (
        logits,
        params,
        token,
    ):
        rt.start(worker)

        setup_tg = rt.task_group()
        rt.fill(of_params.prod(), params, task_group=setup_tg)
        rt.finish_task_group(setup_tg)

        tok_tg = rt.task_group()
        rt.drain(of_token.cons(), token, wait=True, task_group=tok_tg)

        # Three streaming passes over the same DDR logits buffer.
        PINGPONG = 2
        chunk_tgs = []
        for _p in range(N_PASSES):
            for c in range(N_CHUNKS):
                tg = rt.task_group()
                rt.fill(
                    of_logits.prod(),
                    logits,
                    tap=chunk_tap(VOCAB, c * CHUNK_N, CHUNK_N),
                    task_group=tg,
                    wait=True,
                )
                chunk_tgs.append(tg)
                if len(chunk_tgs) > PINGPONG:
                    rt.finish_task_group(chunk_tgs[-1 - PINGPONG])
        for tg in chunk_tgs[max(0, len(chunk_tgs) - PINGPONG):]:
            rt.finish_task_group(tg)
        rt.finish_task_group(tok_tg)

    return Program(NPU2(), rt).resolve_program()


def main():
    argparse.ArgumentParser().parse_args(sys.argv[1:])
    print(build())


if __name__ == "__main__":
    main()
