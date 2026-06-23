"""M3a: standalone fused lm_head GEMM -> 2-memtile logits -> 3-pass full sampler.

Combines M1 (tiled lm_head GEMM at V=128256, 262 MB streamed from DDR) with M2
(streamed temperature/top-k/multinomial sampler), bridged by logits resident
across TWO memtiles (one half each, 256 KB) that the sampler re-reads 3x via
ObjectFifo repeat_count (write-once / read-3x without re-DMA). Validates the
memtile-split + repeat_count choreography in isolation before splicing onto the
decode chain (M3b).

Geometry (validated by aie2_memrepeat_probe.py):
  fp32[V=128256] = 513 KB > one 512 KB memtile -> split into 2 halves of
  HALF=64128 fp32 = 256 KB, each pinned to its own memtile column. The GEMM
  produces 4-logit tiles into chunk fifos; each memtile relay (obj_type=HALF,
  repeat_count=3) ASSEMBLES the chunks into a full half and replays it 3x.

Dataflow:
  xin(int8[D+8]) -> final_norm rms -> normed int8[D]+scale
  normed + lmw(DDR 262 MB) -> lm_head GEMM -> fp32 logits in CHUNK_N chunks,
     routed to half0 (first HALF) then half1 (second HALF)
  relay0/relay1 (memtiles, repeat_count=3) -> sampler: 3 passes x (32 chunks
     from half0 + 32 from half1) -> int32 token

Runtime args: xin(int8[D+8]), gamma(bf16[D]), lmw(int8[WLM]), token(i32[1]) = 4.
Sampler params (temperature/top_k/seed) are a worker-LOCAL Buffer with a baked
initial_value (compile-time per config) so the sampler tile stays within the
2-in/2-out DMA budget (relay0+relay1 in, token out; state+params are local
Buffers, zero DMA). Set via env LLAMA_SAMPLE_TEMP / _TOPK / _SEED. (M3b will
choose a per-token-variable params path from the chain's tile budget.)
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

D = 2048
VOCAB = int(_os.environ.get("LLAMA_SAMPLE_V", "128256"))
N_TILE = 4
N_TILES_LM = VOCAB // N_TILE

HALF = VOCAB // 2  # 64128 fp32 = 256 KB per memtile
CHUNK_N = int(_os.environ.get("LLAMA_SAMPLE_CHUNK", "2004"))
N_CHUNKS = VOCAB // CHUNK_N  # 64
CHUNKS_PER_HALF = HALF // CHUNK_N  # 32
TILES_PER_CHUNK = CHUNK_N // N_TILE  # 501
N_PASSES = 3
STATE_BYTES = 1088

WLM_PREFIX = 64
WLM_SLOT = WLM_PREFIX + N_TILE * D + N_TILE * 4 + N_TILE * 4
WLM_TOTAL = N_TILES_LM * WLM_SLOT

# memtile columns for the two logits halves
MT0, MT1 = Tile(1, 1), Tile(2, 1)

# Baked sampler params (worker-local Buffer initial_value; no DMA channel).
_TEMP = float(_os.environ.get("LLAMA_SAMPLE_TEMP", "0.0"))
_TOPK = int(_os.environ.get("LLAMA_SAMPLE_TOPK", "0"))
_SEED = int(_os.environ.get("LLAMA_SAMPLE_SEED", "0"))


def _params_init():
    import numpy as _np

    tb = _np.frombuffer(_np.float32(_TEMP).tobytes(), dtype=_np.uint32)[0]
    kb = _np.frombuffer(_np.int32(_TOPK).tobytes(), dtype=_np.uint32)[0]
    return _np.asarray([tb, kb, _np.uint32(_SEED & 0xFFFFFFFF)], dtype=_np.uint32)


def _i8(n):
    return np.ndarray[(int(n),), np.dtype[np.int8]]


def _f32(n):
    return np.ndarray[(int(n),), np.dtype[np.float32]]


def _u32(n):
    return np.ndarray[(int(n),), np.dtype[np.uint32]]


def _i32(n):
    return np.ndarray[(int(n),), np.dtype[np.int32]]


def _bf16(n):
    from ml_dtypes import bfloat16

    return np.ndarray[(int(n),), np.dtype[bfloat16]]


def _idx(i):
    return index_cast(IntegerType.get_signless(32), i)


def factor(nb):
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
    rt_xin_ty = _i8(D + 8)
    rt_gamma_ty = _bf16(D)
    rt_lmw_ty = _i8(WLM_TOTAL)
    rt_params_ty = _u32(3)
    rt_token_ty = _i32(1)

    t_xin_i8 = _i8(D + 8)
    t_norm_i8 = _i8(D + 8)
    t_gamma_bf16 = _bf16(D)
    t_wlm_slot = _i8(WLM_SLOT)
    t_chunk_f32 = _f32(CHUNK_N)
    t_half_f32 = _f32(HALF)
    t_state = _i8(STATE_BYTES)
    t_params = _u32(3)
    t_token = _i32(1)

    of_xin = ObjectFifo(t_xin_i8, depth=1, name="xin")
    of_gamma = ObjectFifo(t_gamma_bf16, depth=1, name="gamma")
    of_norm = ObjectFifo(t_norm_i8, depth=1, name="norm")
    of_wlm = ObjectFifo(t_wlm_slot, depth=2, name="wlm")
    of_token = ObjectFifo(t_token, name="token")

    # GEMM produces CHUNK_N-sized logit chunks into two half-streams; each is
    # forwarded through a memtile relay (obj_type=HALF) that assembles the
    # CHUNKS_PER_HALF chunks into a full half and replays it N_PASSES times.
    of_log0 = ObjectFifo(t_chunk_f32, depth=2, name="log0")
    of_log1 = ObjectFifo(t_chunk_f32, depth=2, name="log1")
    of_relay0 = of_log0.cons().forward(
        tile=MT0, obj_type=t_half_f32, depth=1, repeat_count=N_PASSES, name="relay0"
    )
    of_relay1 = of_log1.cons().forward(
        tile=MT1, obj_type=t_half_f32, depth=1, repeat_count=N_PASSES, name="relay1"
    )

    b_state = Buffer(t_state, name="sample_state")
    b_params = Buffer(t_params, initial_value=_params_init(), name="sample_params")

    KO_RMS = "llama_rmsnorm_int8.cc.o"
    KO_GEMM = "llama_gemm_int8_srs_tiled_layer.cc.o"
    KO_SAMP = "llama_sample_streamed.cc.o"

    k_rms = Kernel(
        "llama_rmsnorm_int8_dyn_acttail", KO_RMS, [t_xin_i8, t_gamma_bf16, t_norm_i8]
    )
    k_lmhead = Kernel(
        "llama_gemm_tiled_layer_K2048_N4_lmhead_fp32out",
        KO_GEMM,
        [t_norm_i8, t_wlm_slot, t_chunk_f32, np.int32],
    )
    # Sampler reads the WHOLE half buffer (memtile relay) + a local chunk index.
    k_sample = Kernel(
        "llama_sample_streamed",
        KO_SAMP,
        [t_half_f32, t_state, t_params, np.int32, np.int32, np.int32],
    )
    k_final = Kernel(
        "llama_sample_streamed_finalize", KO_SAMP, [t_state, t_token, t_params]
    )

    def w_rms(c_in, c_gamma, c_out, k):
        x = c_in.acquire(1)
        g = c_gamma.acquire(1)
        o = c_out.acquire(1)
        k(x, g, o)
        c_in.release(1)
        c_gamma.release(1)
        c_out.release(1)

    # GEMM: acquire normed act once; produce all V logits as CHUNK_N chunks,
    # first half0's chunks then half1's. Each chunk = TILES_PER_CHUNK gemm tiles
    # of N_TILE logits (kernel writes out + local_tile*4).
    def w_lmhead(c_act, c_w, c_l0, c_l1, k):
        a = c_act.acquire(1)
        for _ch in range_(CHUNKS_PER_HALF):
            o = c_l0.acquire(1)
            for t in range_(TILES_PER_CHUNK):
                w = c_w.acquire(1)
                k(a, w, o, _idx(t))
                c_w.release(1)
            c_l0.release(1)
        for _ch in range_(CHUNKS_PER_HALF):
            o = c_l1.acquire(1)
            for t in range_(TILES_PER_CHUNK):
                w = c_w.acquire(1)
                k(a, w, o, _idx(t))
                c_w.release(1)
            c_l1.release(1)
        c_act.release(1)

    # Sampler: 3 passes; each pass reads the whole V = half0 (32 chunks via
    # relay0) then half1 (32 chunks via relay1). global chunk index = the
    # sampler's running chunk count within the pass (0..63).
    def w_sample(c0, c1, st, p, c_tok, ks, kf):
        tok = c_tok.acquire(1)
        for pass_i in range_(1, N_PASSES + 1):
            h0 = c0.acquire(1)
            for ch in range_(CHUNKS_PER_HALF):
                ks(h0, st, p, _idx(pass_i), _idx(ch), _idx(ch))
            c0.release(1)
            h1 = c1.acquire(1)
            for ch in range_(CHUNKS_PER_HALF):
                ks(h1, st, p, _idx(pass_i), _idx(CHUNKS_PER_HALF + ch), _idx(ch))
            c1.release(1)
        kf(st, tok, p)
        c_tok.release(1)

    w_rms_worker = Worker(
        w_rms, [of_xin.cons(), of_gamma.cons(), of_norm.prod(), k_rms], tile=Tile(0, 2)
    )
    w_lm_worker = Worker(
        w_lmhead,
        [of_norm.cons(), of_wlm.cons(), of_log0.prod(), of_log1.prod(), k_lmhead],
        tile=Tile(0, 3),
        stack_size=4096,
    )
    w_sample_worker = Worker(
        w_sample,
        [
            of_relay0.cons(),
            of_relay1.cons(),
            b_state,
            b_params,
            of_token.prod(),
            k_sample,
            k_final,
        ],
        tile=Tile(0, 4),
        stack_size=8192,
    )

    rt = Runtime()
    with rt.sequence(rt_xin_ty, rt_gamma_ty, rt_lmw_ty, rt_token_ty) as (
        xin,
        gamma,
        lmw,
        token,
    ):
        rt.start(w_rms_worker, w_lm_worker, w_sample_worker)

        setup_tg = rt.task_group()
        rt.fill(of_xin.prod(), xin, task_group=setup_tg)
        rt.fill(of_gamma.prod(), gamma, task_group=setup_tg)
        rt.finish_task_group(setup_tg)

        tok_tg = rt.task_group()
        rt.drain(of_token.cons(), token, wait=True, task_group=tok_tg)

        PINGPONG = 2
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
    argparse.ArgumentParser().parse_args(sys.argv[1:])
    print(build())


if __name__ == "__main__":
    main()
