"""Standalone tiled lm_head + greedy argmax (M1 of the on-chip sample tile).

Final_norm RMSNorm -> tied lm_head int8 GEMM over the full vocab -> running
greedy argmax -> int32 token id. The 262 MB lm_head weight matrix (V=128256 x
D=2048) streams from DDR tile-by-tile (V/4 = 32064 tiles, depth-2 ping-pong);
the fp32 logits are NEVER materialized -- a fused GEMM+argmax kernel folds each
4-logit tile into a running max held in L1.

Runtime sequence:
   xin    (int8[D+8])     # final hidden: int8 body + fp32 res_scale tail
   lmw    (int8[WLM])     # packed lm_head weights: per-tile slots
                          #   [64 B pad | 4*D i8 weights | 4 i32 bias=0 | 4 fp32 row-scale]
   token  (int32[1])      # output greedy argmax token id

The GEMM's per-token act_scale is read from the normed-activation tail (written
by final_norm rmsnorm_dyn_acttail), NOT from the weight slot prefix.
"""

import argparse
import os as _os
import sys

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern
from aie.dialects.arith import index_cast
from aie.ir import IntegerType

D = 2048
VOCAB = 128256
N_TILE = 4
N_TILES_LM = VOCAB // N_TILE  # 32064

# Weight slot: [64 B pad | N_TILE*D i8 | N_TILE i32 bias | N_TILE fp32 wscale].
WLM_PREFIX = 64
WLM_SLOT = WLM_PREFIX + N_TILE * D + N_TILE * 4 + N_TILE * 4  # 8256
WLM_TOTAL = N_TILES_LM * WLM_SLOT


def _i8(n):
    return np.ndarray[(int(n),), np.dtype[np.int8]]


def _idx(i):
    return index_cast(IntegerType.get_signless(32), i)


def _bf16(n):
    from ml_dtypes import bfloat16

    return np.ndarray[(int(n),), np.dtype[bfloat16]]


def factor(nb):
    """Factor nb into (outer, inner) both <= 1023 (AIE2P BD dim-size limit)."""
    if nb <= 1023:
        return (1, nb)
    for inner in range(min(nb, 1023), 0, -1):
        if nb % inner == 0 and nb // inner <= 1023:
            return (nb // inner, inner)
    raise ValueError(f"can't factor nbytes={nb} into two <=1023 dims")


def strided_tap(total, base_off, per_slot_stride, slot_bytes, n_slots):
    outer, inner = factor(slot_bytes)
    return TensorAccessPattern(
        tensor_dims=[total],
        offset=base_off,
        sizes=[1, n_slots, outer, inner],
        strides=[0, per_slot_stride, inner, 1],
    )


def build():
    rt_xin_ty = _i8(D + 8)  # hidden int8[D] + fp32 res_scale + pad
    rt_gamma_ty = _bf16(D)  # final_norm gamma
    rt_lmw_ty = _i8(WLM_TOTAL)
    rt_token_ty = _i8(8)  # [best_idx i32 | best_val fp32]; host reads idx

    t_xin_i8 = _i8(D + 8)
    t_norm_i8 = _i8(D + 8)  # final_norm out: int8[D] + fp32 scale tail
    t_gamma_bf16 = _bf16(D)
    t_wlm_slot = _i8(WLM_SLOT)
    t_state_i8 = _i8(8)  # [best_idx i32 | best_val fp32]

    of_xin = ObjectFifo(t_xin_i8, depth=1, name="xin")
    of_gamma = ObjectFifo(t_gamma_bf16, depth=1, name="gamma")
    of_norm = ObjectFifo(t_norm_i8, depth=1, name="norm")
    of_wlm = ObjectFifo(t_wlm_slot, depth=2, name="wlm")
    of_token = ObjectFifo(t_state_i8, name="token")

    KO_RMS = "llama_rmsnorm_int8.cc.o"
    KO_GEMM = "llama_gemm_int8_srs_tiled_layer.cc.o"

    k_rms = Kernel(
        "llama_rmsnorm_int8_dyn_acttail", KO_RMS, [t_xin_i8, t_gamma_bf16, t_norm_i8]
    )
    k_lmhead = Kernel(
        "llama_gemm_tiled_layer_K2048_N4_lmhead_argmax",
        KO_GEMM,
        [t_norm_i8, t_wlm_slot, t_state_i8, np.int32],
    )

    def w_rms(c_in, c_gamma, c_out, k):
        x = c_in.acquire(1)
        g = c_gamma.acquire(1)
        o = c_out.acquire(1)
        k(x, g, o)
        c_in.release(1)
        c_gamma.release(1)
        c_out.release(1)

    # Fused lm_head GEMM + running argmax. Acquire the normed activation +
    # token output once; loop all 32064 weight tiles, each folding 4 logits
    # into the running max. The kernel self-seeds at tile_idx==0 and writes
    # the final best_idx to state[0] (reinterpreted as the int32 token).
    def w_lmhead(c_act, c_w, c_tok, k):
        a = c_act.acquire(1)
        state = c_tok.acquire(1)
        for t in range_(N_TILES_LM):
            w = c_w.acquire(1)
            k(a, w, state, _idx(t))
            c_w.release(1)
        c_act.release(1)
        c_tok.release(1)

    w_rms_worker = Worker(
        w_rms, [of_xin.cons(), of_gamma.cons(), of_norm.prod(), k_rms], tile=Tile(5, 4)
    )
    # stack: kernel is light (no big arrays); default is fine, bump for margin.
    w_lm_worker = Worker(
        w_lmhead,
        [of_norm.cons(), of_wlm.cons(), of_token.prod(), k_lmhead],
        tile=Tile(5, 5),
        stack_size=4096,
    )

    rt = Runtime()
    with rt.sequence(rt_xin_ty, rt_gamma_ty, rt_lmw_ty, rt_token_ty) as (
        xin,
        gamma,
        lmw,
        token,
    ):
        rt.start(w_rms_worker, w_lm_worker)

        xin_tg = rt.task_group()
        rt.fill(of_xin.prod(), xin, task_group=xin_tg)
        rt.fill(of_gamma.prod(), gamma, task_group=xin_tg)
        rt.finish_task_group(xin_tg)

        tok_tg = rt.task_group()
        rt.drain(of_token.cons(), token, wait=True, task_group=tok_tg)

        # Stream the 262 MB lm_head weights, depth-2 ping-pong over tiles.
        PINGPONG = int(_os.environ.get("LLAMA_LM_PINGPONG", "2"))
        tile_tgs = [rt.task_group() for _ in range(N_TILES_LM)]
        for t in range(N_TILES_LM):
            rt.fill(
                of_wlm.prod(),
                lmw,
                tap=strided_tap(WLM_TOTAL, t * WLM_SLOT, WLM_SLOT, WLM_SLOT, 1),
                task_group=tile_tgs[t],
                wait=True,
            )
            if t >= PINGPONG:
                rt.finish_task_group(tile_tgs[t - PINGPONG])
        for t in range(max(0, N_TILES_LM - PINGPONG), N_TILES_LM):
            rt.finish_task_group(tile_tgs[t])
        rt.finish_task_group(tok_tg)

    return Program(NPU2(), rt).resolve_program()


def main():
    p = argparse.ArgumentParser()
    p.parse_args(sys.argv[1:])
    print(build())


if __name__ == "__main__":
    main()
