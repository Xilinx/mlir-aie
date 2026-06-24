"""Phase 7b: N-layer decode chain at PRODUCTION multi-head GQA shapes
(32 Q heads, 8 KV heads, REP=4, HEAD_DIM=64, Q_DIM=2048, KV_DIM=512,
HD=8192, T=128).

Combines:
  - aie2_layer_mh.py topology (q_proj_mh + rope_mh + q_split + 8 attn +
    sv_merge + af_concat + o_proj_mh + FFN-half, plus af_scales fifo)
  - aie2_chain_dynscale.py multi-layer wrapper (residual loop-back via
    router, range_(N_LAYERS) in each worker, per-fifo per-layer fill
    with ping-pong BD reuse).

All ObjectFifos depth=1 (Bug 8). FFN-side stack_size matches
layer_mh's PSK=8192 (Bug 12 -- larger silently corrupts).
"""

import argparse
import os as _os
import sys

import numpy as np

from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern
from aie.dialects.arith import index_cast, constant
from aie.ir import IntegerType

# M3b: fuse final_norm + lm_head GEMM + memtile logits relay + streamed sampler
# onto the chain's final output, producing an int32 token (host-free decode).
# Guarded so the default chain stays byte-identical.
SAMPLE = _os.environ.get("LLAMA_CHAIN_SAMPLE", "0") == "1"
# M3c: on-chip embed gather at the FRONT -- the dispatch input becomes a token
# id (int32); a gather worker selects embed[token] from the lm_head weight
# stream and produces the layer-0 seed. Makes the dispatch a token->token
# function (the shape the persistent loop needs). Requires SAMPLE.
GATHER = _os.environ.get("LLAMA_CHAIN_GATHER", "0") == "1" and SAMPLE
# One-stream sampler + embed gather: the fused lm_head GEMM + top-k insert runs
# over a SINGLE 262 MB table pass and emits BOTH the token and the next-token
# embed seed (no relay, no 3-pass replay, no second gather stream). Replaces the
# SAMPLE relay path; mutually exclusive with it. The shape the persistent loop
# needs (token in -> token + seed out). When LLAMA_CHAIN_ONESTREAM=1.
ONESTREAM = _os.environ.get("LLAMA_CHAIN_ONESTREAM", "0") == "1" and SAMPLE
ONESTREAM_KSET = int(_os.environ.get("LLAMA_CHAIN_ONESTREAM_KSET", "16"))
# Persistent on-device autoregressive loop (capstone): PT tokens generated in ONE
# dispatch, the sampled token's embed seed feeding back to the chain ON-CHIP (no
# host between tokens). Requires ONESTREAM. Increment 1: KV held at a fixed
# position (host re-supplies the same KV per token); proves the device-originated
# control loop bit-exact. When LLAMA_CHAIN_PERSIST=1.
PERSIST = _os.environ.get("LLAMA_CHAIN_PERSIST", "0") == "1" and ONESTREAM
# Growing KV (increment 2): the KV cache accumulates across the PT tokens (real
# autoregressive decode) via the on-chip advancing append + host KV ping-pong.
# When off (increment 1), KV is held at a fixed position (host re-streams pristine
# KV each token). Requires PERSIST.
PERSIST_GROW = _os.environ.get("LLAMA_CHAIN_PERSIST_GROW", "0") == "1" and PERSIST
# Resident KV (increment toward 100% NPU): the KV cache BODY lives in worker-local
# buffers on each attn tile (N_LAYERS caches per head), seeded once from the host
# on token 0 then read-modify-written IN PLACE -- zero KV DMA after the seed (host
# streams ONLY weights). Requires PERSIST_GROW. Scoped to small N_LAYERS that fit
# the compute tile (N=2 -> 32KB/head); N=16 (256KB/head) is a memtile follow-up.
PERSIST_RESIDENT = (
    _os.environ.get("LLAMA_CHAIN_PERSIST_RESIDENT", "0") == "1" and PERSIST_GROW
)
# Tokens per dispatch. PT==1 (default off) makes every PERSIST-aware loop a no-op,
# so the non-persist paths stay byte-identical.
PT = int(_os.environ.get("LLAMA_CHAIN_PT", "4")) if PERSIST else 1
if SAMPLE and not ONESTREAM:
    from logits_relay import LogitsRelay


def _sample_params_init():
    tb = np.frombuffer(np.float32(SAMPLE_TEMP).tobytes(), dtype=np.uint32)[0]
    kb = np.frombuffer(np.int32(SAMPLE_TOPK).tobytes(), dtype=np.uint32)[0]
    return np.asarray([tb, kb, np.uint32(SAMPLE_SEED & 0xFFFFFFFF)], dtype=np.uint32)


def _onestream_params_init():
    # int32-packed [temperature bits | top_k | seed] -- NOT uint32 (the IRON
    # ui32 Buffer initial_value silently zeros large words; the finalize kernel
    # memcpy's the bytes so signedness is irrelevant). Reuses the SAMPLE_* env.
    tb = np.frombuffer(np.float32(SAMPLE_TEMP).tobytes(), dtype=np.int32)[0]
    sb = np.frombuffer(np.uint32(SAMPLE_SEED & 0xFFFFFFFF).tobytes(), dtype=np.int32)[0]
    return np.asarray([tb, np.int32(SAMPLE_TOPK), sb], dtype=np.int32)


def _i32(idx):
    return index_cast(IntegerType.get_signless(32), idx)


def _const_i32(v):
    return constant(IntegerType.get_signless(32), v)


# --- Production multi-head shapes ---
D = 2048
HD = 8192
HEAD_D = 64
N_HEADS_Q = 32
N_HEADS_KV = 8
REP = N_HEADS_Q // N_HEADS_KV  # 4
QD = N_HEADS_Q * HEAD_D  # 2048
KV_DIM = N_HEADS_KV * HEAD_D  # 512 (k_proj / v_proj output dim)
T = int(_os.environ.get("LLAMA_CHAIN_T", "128"))
N_TILE = 4
N_LAYERS = int(_os.environ.get("LLAMA_CHAIN_N", "2"))

# --- M3b sample-tile constants (only used when SAMPLE) ---
VOCAB = 128256
LM_N_TILES = VOCAB // N_TILE  # 32064
WLM_PREFIX = 64
WLM_SLOT = WLM_PREFIX + N_TILE * D + N_TILE * 4 + N_TILE * 4  # 8256
WLM_TOTAL = LM_N_TILES * WLM_SLOT  # 262 MB
SAMPLE_CHUNK = 2004  # V = 64 * 2004
SAMPLE_N_CHUNKS = VOCAB // SAMPLE_CHUNK  # 64
SAMPLE_TILES_PER_CHUNK = SAMPLE_CHUNK // N_TILE  # 501
SAMPLE_PASSES = 3
SAMPLE_STATE_BYTES = 1088
# Baked sampler params (worker-local Buffer; 0 DMA). Greedy by default.
SAMPLE_TEMP = float(_os.environ.get("LLAMA_SAMPLE_TEMP", "0.0"))
SAMPLE_TOPK = int(_os.environ.get("LLAMA_SAMPLE_TOPK", "0"))
SAMPLE_SEED = int(_os.environ.get("LLAMA_SAMPLE_SEED", "0"))

# Per-Q-head tail: 32 * [q_out_scale fp32, sv_inv_out_scale fp32] = 256 B.
QF_TAIL = N_HEADS_Q * 8
QF_BYTES = QD + QF_TAIL  # 2304
QR_BYTES = QF_BYTES
QCHUNK_BODY = REP * HEAD_D  # 256
QCHUNK_TAIL = REP * 8  # 32
QCHUNK_BYTES = QCHUNK_BODY + QCHUNK_TAIL  # 288
QCHUNKS_HALF_BYTES = (N_HEADS_KV // 2) * QCHUNK_BYTES  # 1152

# On-chip KV append (mirrors aie2_layer_mh): k_proj/v_proj emit fp32[KV_DIM];
# fp32 outputs + cos/sin join at a memtile into kvcs; qkv_combine builds
# per-head combined chunks [q_chunk 288 | k_fp 256 | v_fp 256 | cs 256].
KFP_HEAD = HEAD_D * 4  # 256 B (64 fp32) per KV head
KFP_ALL = N_HEADS_KV * KFP_HEAD  # 2048
CS_PACK_BYTES = HEAD_D * 2 * 2  # 256 (cos+sin bf16), shared across heads
KVCS_BYTES = 2 * KFP_ALL + CS_PACK_BYTES  # 4352
COMB_CHUNK_BYTES = QCHUNK_BYTES + 2 * KFP_HEAD + CS_PACK_BYTES  # 1056
COMB_HALF_BYTES = (N_HEADS_KV // 2) * COMB_CHUNK_BYTES  # 4224

# attn worker output: REP heads of HEAD_D body + REP per-head sv_out_scale
# (self-cal 1c): [REP*HEAD_D body | REP*4 scale].
SVCHUNK_BODY = REP * HEAD_D  # 256
SVCHUNK_TAIL = REP * 4  # 16
SVCHUNK_BYTES = SVCHUNK_BODY + SVCHUNK_TAIL  # 272
SV_CONCAT_HALF = (N_HEADS_KV // 2) * SVCHUNK_BYTES  # 1088
AF_BYTES = N_HEADS_Q * HEAD_D  # 2048
# sv_merge_selfcal output: [AF_BYTES bodies | N_HEADS_Q*4 sv_out_scales].
SVFULL_BYTES = AF_BYTES + N_HEADS_Q * 4  # 2176

# Per-slot KV: each cached position carries its OWN k/v scale (fixes the
# per-head-scalar bug). Header is T fp32 per-slot scales.
KV_HEADER = T * 4
KCACHE_BYTES = T * HEAD_D  # 8192
VCACHE_BYTES = T * HEAD_D  # 8192
KCACHE_PADDED = KV_HEADER + KCACHE_BYTES
VCACHE_PADDED = KV_HEADER + VCACHE_BYTES

AF_SCALES_BYTES = 192

PREFIX_ALIGN = 64
WQ_PREFIX = 448
WO_PREFIX = PREFIX_ALIGN
WG_PREFIX = 0
WU_PREFIX = PREFIX_ALIGN
WD_PREFIX = PREFIX_ALIGN
assert WQ_PREFIX % 64 == 0

# k_proj/v_proj prefix: fp32out_acttail reads act_scale from the h1 tail, so
# only the 64 B alignment pad is needed.
WK_PREFIX = PREFIX_ALIGN
WV_PREFIX = PREFIX_ALIGN

WQ_SLOT = WQ_PREFIX + N_TILE * D + N_TILE * 4 + N_TILE * 4
WO_SLOT = WO_PREFIX + N_TILE * QD + N_TILE * 4 + N_TILE * 4
WG_SLOT = WG_PREFIX + N_TILE * D + N_TILE * 4 + N_TILE * 4
WU_SLOT = WU_PREFIX + N_TILE * D + N_TILE * 4 + N_TILE * 4
WD_SLOT = WD_PREFIX + N_TILE * HD + N_TILE * 4 + N_TILE * 4
WK_SLOT = WK_PREFIX + N_TILE * D + N_TILE * 4 + N_TILE * 4
WV_SLOT = WV_PREFIX + N_TILE * D + N_TILE * 4 + N_TILE * 4

N_TILES_Q = QD // N_TILE
N_TILES_O = D // N_TILE
N_TILES_G = HD // N_TILE
N_TILES_U = N_TILES_G
N_TILES_D = D // N_TILE
N_TILES_K = KV_DIM // N_TILE  # 128
N_TILES_V = N_TILES_K

GAMMA_BYTES = D * 2
CS_BYTES = HEAD_D * 2 * 2

# Per-fifo total bytes (contiguous across N_LAYERS in wblob).
GAMMA_TOTAL = N_LAYERS * GAMMA_BYTES
CS_TOTAL = N_LAYERS * CS_BYTES
WQ_TOTAL = N_LAYERS * N_TILES_Q * WQ_SLOT
WO_TOTAL = N_LAYERS * N_TILES_O * WO_SLOT
WG_TOTAL = N_LAYERS * N_TILES_G * WG_SLOT
WU_TOTAL = N_LAYERS * N_TILES_U * WU_SLOT
WD_TOTAL = N_LAYERS * N_TILES_D * WD_SLOT
WK_TOTAL = N_LAYERS * N_TILES_K * WK_SLOT
WV_TOTAL = N_LAYERS * N_TILES_V * WV_SLOT
AF_SCALES_TOTAL = N_LAYERS * AF_SCALES_BYTES

OFF_GAMMA_IN = 0
OFF_WQ = OFF_GAMMA_IN + GAMMA_TOTAL
OFF_CS = OFF_WQ + WQ_TOTAL
OFF_WO = OFF_CS + CS_TOTAL
OFF_AF_SCALES = OFF_WO + WO_TOTAL
OFF_GAMMA_POST = OFF_AF_SCALES + AF_SCALES_TOTAL
OFF_WG = OFF_GAMMA_POST + GAMMA_TOTAL
OFF_WU = OFF_WG + WG_TOTAL
OFF_WD = OFF_WU + WU_TOTAL
OFF_WK = OFF_WD + WD_TOTAL
OFF_WV = OFF_WK + WK_TOTAL
WEIGHTS_BYTES = OFF_WV + WV_TOTAL

# PERSIST_GROW per-position rope: each of the PT tokens generates at position
# P0+t, which needs ITS OWN cos/sin (the rotary phase advances with position).
# Append a PT*N_LAYERS*CS_BYTES block to wblob; the of_cs (q-rope) + kvcs_eps[2]
# (append rope_k) fills read it at the (tok,L) offset. CS_BYTES=256 so the whole
# block is tiny (PT*N_LAYERS*256). Keeps the runtime at 5 args (folded into the
# existing wblob). Laid out [token0: L0..L_{N-1} | token1: ... | ...].
# Always define these at module scope (importable regardless of the env flag);
# only EXTEND WEIGHTS_BYTES with the per-position block when PERSIST_GROW, so the
# non-grow wblob size is unchanged.
OFF_CS_PERPOS = WEIGHTS_BYTES
CS_PERPOS_TOTAL = PT * N_LAYERS * CS_BYTES
if PERSIST_GROW:
    WEIGHTS_BYTES = OFF_CS_PERPOS + CS_PERPOS_TOTAL

# kvblob: per-layer block, each containing 8 KV heads' (T_used | k_header |
# kcache | v_header | vcache) combined slots. Phase 8c adds the 4-byte
# T_used prefix per KV head -> per-head slot grows from 16392 to 16396 B.
T_USED_BYTES = 8  # 4 B T_used + 4 B pad -> 16400 B per-head (factorable)
PER_KV_HEAD_BYTES = T_USED_BYTES + KCACHE_PADDED + VCACHE_PADDED  # 16400
PER_LAYER_KV = N_HEADS_KV * PER_KV_HEAD_BYTES  # 131168
KV_HALF_BYTES = (N_HEADS_KV // 2) * PER_KV_HEAD_BYTES  # 65584
KV_BYTES = N_LAYERS * PER_LAYER_KV


def _envf(name, default):
    return float(_os.environ.get(name, default))


ACT_SCALE = _envf("LAYER_ACT_SCALE", "0.05")
INV_ACT_SCALE = _envf("LAYER_INV_ACT_SCALE", str(1.0 / 0.05))
SILU_GATE_SCALE = _envf("SILU_GATE_SCALE", "0.05")
GATE_INV_OUT_SCALE = 1.0 / SILU_GATE_SCALE


def _i8(n):
    return np.ndarray[(int(n),), np.dtype[np.int8]]


def _bf16(n):
    from ml_dtypes import bfloat16

    return np.ndarray[(int(n),), np.dtype[bfloat16]]


def _f32(n):
    return np.ndarray[(int(n),), np.dtype[np.float32]]


def _u32(n):
    return np.ndarray[(int(n),), np.dtype[np.uint32]]


def _i32a(n):
    return np.ndarray[(int(n),), np.dtype[np.int32]]


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
    # Residual stream carries a per-token dynamic scale in an int8[D+8] tail
    # (Phase B residual_dyn): xin seed + layer out are both int8[D+8].
    rt_xin_ty = _i8(D + 8)
    rt_w_ty = _i8(WEIGHTS_BYTES)
    # PERSIST_RESIDENT: KV lives on-chip (b_kv); the host supplies only the 1x
    # pristine seed (filled once on token 0). PERSIST (host-ferried) uses a 2x
    # [pristine | scratch] (fixed-pos) or ping-pong (grow) buffer. Non-persist: 1x.
    rt_kv_ty = _i8(2 * KV_BYTES if (PERSIST and not PERSIST_RESIDENT) else KV_BYTES)
    rt_out_ty = _i8(D + 8)

    # M3b: lmw blob = [final_norm gamma bf16(D) = 2*D bytes | lm_head weight
    # slots]. token output replaces the host hidden drain.
    LMW_GAMMA_BYTES = 2 * D  # bf16[D]
    rt_lmw_ty = _i8(LMW_GAMMA_BYTES + WLM_TOTAL)
    rt_token_ty = _i32a(1)

    t_D_i8 = _i8(D)
    t_D8_i8 = _i8(D + 8)  # residual / rmsnorm-dyn: D int8 + 4 B scale + 4 B pad
    t_D_f32 = _f32(D)  # o_proj / down fp32 output -> rescale-add
    t_QF_i8 = _i8(QF_BYTES)
    t_QR_i8 = _i8(QR_BYTES)
    t_QCHUNKS_HALF_i8 = _i8(QCHUNKS_HALF_BYTES)
    t_QCHUNK_i8 = _i8(QCHUNK_BYTES)
    t_SVCHUNK_i8 = _i8(SVCHUNK_BYTES)
    t_SVCONCAT_HALF_i8 = _i8(SV_CONCAT_HALF)
    t_AFSCALES_i8 = _i8(AF_SCALES_BYTES)
    t_AF_i8 = _i8(AF_BYTES)
    t_AF8_i8 = _i8(AF_BYTES + 8)  # af + self-cal o_act tail (1b)
    t_HD_i8 = _i8(HD)
    t_UF_i8 = _i8(HD + 8)
    t_D_bf16 = _bf16(D)
    t_CS_bf16 = _bf16(HEAD_D * 2)
    t_WQ_slot = _i8(WQ_SLOT)
    t_WO_slot = _i8(WO_SLOT)
    t_WG_slot = _i8(WG_SLOT)
    t_WU_slot = _i8(WU_SLOT)
    t_WD_slot = _i8(WD_SLOT)
    t_WK_slot = _i8(WK_SLOT)
    t_WV_slot = _i8(WV_SLOT)
    t_KV_i8 = _i8(PER_KV_HEAD_BYTES)
    t_KV_HALF_i8 = _i8(KV_HALF_BYTES)
    t_KVfp = _f32(KV_DIM)
    t_KVCS_i8 = _i8(KVCS_BYTES)
    t_COMB_i8 = _i8(COMB_CHUNK_BYTES)
    t_COMB_HALF_i8 = _i8(COMB_HALF_BYTES)

    # --- ObjectFifos (all depth=1) ---
    # Residual loop-back. All carry the per-token scale tail (int8[D+8]).
    of_seed = ObjectFifo(t_D8_i8, depth=1, name="seed")
    of_back = ObjectFifo(t_D8_i8, depth=1, name="back")
    of_routed = ObjectFifo(t_D8_i8, depth=1, name="routed")  # rmsnorm1 + add1
    of_out = ObjectFifo(t_D8_i8, depth=1, name="layer_out")
    if PERSIST:
        # Persistent loop: token 0's layer-0 input from the host (of_hostseed),
        # tokens 1..PT-1 from the on-chip feedback (of_feedback, depth>=2 self-
        # feedback across the onestream->seedmux tile boundary). seedmux selects
        # between them and drives of_seed (the router's input).
        of_hostseed = ObjectFifo(t_D8_i8, depth=1, name="hostseed")
        of_feedback = ObjectFifo(t_D8_i8, depth=2, name="feedback")

    # M3b sample block: final hidden (of_out) -> final_norm rms -> lm_head GEMM
    # -> memtile logits relay -> streamed sampler -> token.
    if SAMPLE:
        t_lm_chunk_f32 = _f32(SAMPLE_CHUNK)
        t_wlm_slot = _i8(WLM_SLOT)
        t_state_i8 = _i8(SAMPLE_STATE_BYTES)
        t_params_u32 = _u32(3)
        of_fnorm = ObjectFifo(t_D8_i8, depth=1, name="fnorm")  # final_norm out
        of_fgam = ObjectFifo(t_D_bf16, depth=1, name="fgam")  # final_norm gamma
        of_wlm = ObjectFifo(t_wlm_slot, depth=2, name="wlm")
        of_token = ObjectFifo(_i32a(1), name="token")
        if ONESTREAM:
            # One-stream: fused GEMM+insert over of_wlm -> finalize emits token +
            # next-token embed seed. Resident top-k set held in worker-local
            # Buffers (capacity ONESTREAM_KSET). params packed int32 (the ui32
            # Buffer-init bug zeros large words -- see iron-ui32-buffer-init-bug).
            # Packed output [seed int8[D] | scale f32 | token i32 | pad]; one
            # fifo keeps the runtime at 5 args (run_test segfaults at ~6).
            t_opacked = _i8(D + 12)
            of_opacked = ObjectFifo(t_opacked, name="opacked")
            KS = ONESTREAM_KSET
            b_set_logit = Buffer(_f32(KS), name="os_set_logit")
            b_set_gidx = Buffer(_i32a(KS), name="os_set_gidx")
            b_set_scale = Buffer(_f32(KS), name="os_set_scale")
            b_set_row = Buffer(_i8(KS * D), name="os_set_row")
            b_set_len = Buffer(
                _i32a(1), initial_value=np.zeros(1, np.int32), name="os_set_len"
            )
            b_os_params = Buffer(
                _i32a(3),
                initial_value=_onestream_params_init(),
                name="os_params",
            )
        else:
            lm_relay = LogitsRelay(
                total_elems=VOCAB,
                chunk_elems=SAMPLE_CHUNK,
                repeat_count=SAMPLE_PASSES,
                memtile_placement=Tile(4, 1),
                gemm_placement=Tile(5, 4),
                sampler_placement=Tile(5, 5),
                name="lmlogits",
            )
            b_sample_state = Buffer(t_state_i8, name="sample_state")
            b_sample_params = Buffer(
                t_params_u32, initial_value=_sample_params_init(), name="sample_params"
            )

    if GATHER:
        # Front gather: a second pass over the lm_head weight slots; the gather
        # worker selects embed[token_in] -> the layer-0 seed (of_seed).
        of_gwlm = ObjectFifo(t_wlm_slot, depth=2, name="gwlm")
        # token_in: the input token id, host-supplied (this step) -> on-chip
        # (capstone). Worker-local Buffer, host-filled via a tiny runtime arg.
        of_token_in = ObjectFifo(_i32a(1), depth=1, name="token_in")

    of_gam_in = ObjectFifo(t_D_bf16, depth=1, name="gam_in")
    of_h1 = ObjectFifo(t_D8_i8, depth=1, name="h1")
    of_wq = ObjectFifo(t_WQ_slot, depth=2, name="wq")
    of_wk = ObjectFifo(t_WK_slot, depth=2, name="wk")
    of_wv = ObjectFifo(t_WV_slot, depth=2, name="wv")
    of_qfp = ObjectFifo(_f32(QD), depth=1, name="qfp")  # fp32 q_proj out (2a)
    of_qf = ObjectFifo(t_QF_i8, depth=1, name="qf")
    of_cs = ObjectFifo(t_CS_bf16, depth=1, name="cs")
    of_qr = ObjectFifo(t_QR_i8, depth=1, name="qr")

    # On-chip KV append: k_proj/v_proj fp32 outputs + cos/sin join at a memtile
    # into kvcs [kfp 2048 | vfp 2048 | cs 256]; qkv_combine builds per-head
    # combined chunks fed to co-located append+flowkv tiles.
    of_kvcs = ObjectFifo(t_KVCS_i8, depth=1, name="kvcs")
    kvcs_eps = of_kvcs.prod().join(
        offsets=[0, KFP_ALL, 2 * KFP_ALL],
        obj_types=[t_KVfp, t_KVfp, _i8(CS_PACK_BYTES)],
        names=["kvcs_k", "kvcs_v", "kvcs_cs"],
    )

    of_comb_lo = ObjectFifo(t_COMB_HALF_i8, depth=1, name="comb_lo")
    of_comb_hi = ObjectFifo(t_COMB_HALF_i8, depth=1, name="comb_hi")
    comb_lo_eps = of_comb_lo.cons().split(
        offsets=[i * COMB_CHUNK_BYTES for i in range(N_HEADS_KV // 2)],
        obj_types=[t_COMB_i8] * (N_HEADS_KV // 2),
        names=[f"comb_{i}" for i in range(N_HEADS_KV // 2)],
    )
    comb_hi_eps = of_comb_hi.cons().split(
        offsets=[i * COMB_CHUNK_BYTES for i in range(N_HEADS_KV // 2)],
        obj_types=[t_COMB_i8] * (N_HEADS_KV // 2),
        names=[f"comb_{N_HEADS_KV // 2 + i}" for i in range(N_HEADS_KV // 2)],
    )
    comb_eps = list(comb_lo_eps) + list(comb_hi_eps)

    of_kv_lo = ObjectFifo(t_KV_HALF_i8, depth=1, name="kv_lo")
    of_kv_hi = ObjectFifo(t_KV_HALF_i8, depth=1, name="kv_hi")
    kv_lo_eps = of_kv_lo.cons().split(
        offsets=[i * PER_KV_HEAD_BYTES for i in range(N_HEADS_KV // 2)],
        obj_types=[t_KV_i8] * (N_HEADS_KV // 2),
        names=[f"kv_{i}" for i in range(N_HEADS_KV // 2)],
    )
    kv_hi_eps = of_kv_hi.cons().split(
        offsets=[i * PER_KV_HEAD_BYTES for i in range(N_HEADS_KV // 2)],
        obj_types=[t_KV_i8] * (N_HEADS_KV // 2),
        names=[f"kv_{N_HEADS_KV // 2 + i}" for i in range(N_HEADS_KV // 2)],
    )
    of_kvs = list(kv_lo_eps) + list(kv_hi_eps)

    # Device-owned cache drain: 8 updated head caches join into 2 halves.
    # PERSIST_RESIDENT keeps the cache on-chip (worker-local b_kv) -> no drain.
    if not PERSIST_RESIDENT:
        of_kvout_lo = ObjectFifo(t_KV_HALF_i8, depth=1, name="kvout_lo")
        of_kvout_hi = ObjectFifo(t_KV_HALF_i8, depth=1, name="kvout_hi")
        kvout_lo_eps = of_kvout_lo.prod().join(
            offsets=[i * PER_KV_HEAD_BYTES for i in range(N_HEADS_KV // 2)],
            obj_types=[t_KV_i8] * (N_HEADS_KV // 2),
            names=[f"kvout_{i}" for i in range(N_HEADS_KV // 2)],
        )
        kvout_hi_eps = of_kvout_hi.prod().join(
            offsets=[i * PER_KV_HEAD_BYTES for i in range(N_HEADS_KV // 2)],
            obj_types=[t_KV_i8] * (N_HEADS_KV // 2),
            names=[f"kvout_{N_HEADS_KV // 2 + i}" for i in range(N_HEADS_KV // 2)],
        )
        kvout_eps = list(kvout_lo_eps) + list(kvout_hi_eps)

    if PERSIST_RESIDENT:
        # Per-attn-head resident KV: N_LAYERS caches each, held on the attn tile
        # across the whole dispatch (read-modify-write in place). 0-DMA Buffers.
        b_kv = [
            Buffer(
                _i8(N_LAYERS * PER_KV_HEAD_BYTES),
                initial_value=np.zeros(N_LAYERS * PER_KV_HEAD_BYTES, np.int8),
                name=f"kv_resident_{i}",
            )
            for i in range(N_HEADS_KV)
        ]

    of_svconcat_lo = ObjectFifo(t_SVCONCAT_HALF_i8, depth=1, name="sv_concat_lo")
    of_svconcat_hi = ObjectFifo(t_SVCONCAT_HALF_i8, depth=1, name="sv_concat_hi")
    svchunk_lo_eps = of_svconcat_lo.prod().join(
        offsets=[i * SVCHUNK_BYTES for i in range(N_HEADS_KV // 2)],
        obj_types=[t_SVCHUNK_i8] * (N_HEADS_KV // 2),
        names=[f"svchunk_{i}" for i in range(N_HEADS_KV // 2)],
    )
    svchunk_hi_eps = of_svconcat_hi.prod().join(
        offsets=[i * SVCHUNK_BYTES for i in range(N_HEADS_KV // 2)],
        obj_types=[t_SVCHUNK_i8] * (N_HEADS_KV // 2),
        names=[f"svchunk_{N_HEADS_KV // 2 + i}" for i in range(N_HEADS_KV // 2)],
    )
    svchunk_eps = list(svchunk_lo_eps) + list(svchunk_hi_eps)

    of_svfull = ObjectFifo(_i8(SVFULL_BYTES), depth=1, name="sv_full")  # bodies+scales
    # of_afscales removed: sv_out_scales self-calibrated (1c) via merged sv tail;
    # o_act_scale self-calibrated (1b) via af tail.
    of_af = ObjectFifo(t_AF8_i8, depth=1, name="af")  # AF+8: self-cal o_act tail
    of_wo = ObjectFifo(t_WO_slot, depth=2, name="wo")
    of_op = ObjectFifo(t_D_f32, depth=1, name="op")  # fp32 o_proj output

    of_x1 = ObjectFifo(t_D8_i8, depth=1, name="x1")  # residual int8[D+8]

    of_gam_post = ObjectFifo(t_D_bf16, depth=1, name="gam_post")
    of_h2 = ObjectFifo(t_D8_i8, depth=1, name="h2")
    of_wg = ObjectFifo(t_WG_slot, depth=2, name="wg")
    # depth=1 (not 2 like sibling weight fifos): self-cal adds of_ufp
    # (fp32[HD]=32KB) on the up tile (5,2); double-buffering of_wu too overflows
    # L1 there -> corruption (N=2 0/4 with depth=2). depth=1 fits + passes.
    of_wu = ObjectFifo(t_WU_slot, depth=1, name="wu")
    of_wd = ObjectFifo(t_WD_slot, depth=1, name="wd")  # K=8192 L1 budget
    of_gf = ObjectFifo(t_HD_i8, depth=1, name="gf")
    of_ufp = ObjectFifo(_f32(HD), depth=1, name="ufp")  # fp32 up_proj out (2b)
    of_uf = ObjectFifo(t_UF_i8, depth=1, name="uf")  # int8 up + up_out_scale tail
    of_sf = ObjectFifo(t_UF_i8, depth=1, name="sf")  # HD+8: self-cal silu_out tail
    of_df = ObjectFifo(t_D_f32, depth=1, name="df")  # fp32 down output

    KO_RMS = "llama_rmsnorm_int8.cc.o"
    KO_GEMM = "llama_gemm_int8_srs_tiled_layer.cc.o"
    KO_GEMM2 = "llama_gemm_int8_srs_tiled_layer_mh.cc.o"
    KO_ROPE = "llama_rope_int8_mh.cc.o"
    KO_SILU = "llama_silu_mul_int8.cc.o"
    KO_FKV = "llama_flowkv_mh.cc.o"
    KO_GLUE = "llama_gqa_glue.cc.o"
    KO_PT = "llama_layer_pt.cc.o"
    KO_SAMP = "llama_sample_streamed.cc.o"

    # rmsnorm acttail: per-token input scale read from the residual tail.
    k_rms = Kernel(
        "llama_rmsnorm_int8_dyn_acttail", KO_RMS, [t_D8_i8, t_D_bf16, t_D8_i8]
    )

    if SAMPLE and ONESTREAM:
        # One-stream: fused lm_head GEMM + top-k insert, then finalize -> token +
        # next-token embed seed. final_norm reuses k_rms (below).
        KS = ONESTREAM_KSET
        k_os_insert = Kernel(
            "llama_lmhead_topk_insert",
            KO_GEMM,
            [
                t_D8_i8,
                t_wlm_slot,
                _f32(KS),
                _i32a(KS),
                _f32(KS),
                _i8(KS * D),
                _i32a(1),
                np.int32,
            ],
        )
        k_os_final = Kernel(
            "llama_topk_finalize_packed",
            "llama_topk_sample.cc.o",
            [
                _f32(KS),
                _i32a(KS),
                _f32(KS),
                _i8(KS * D),
                _i32a(1),
                _i32a(3),
                _i8(D + 12),
            ],
        )
    elif SAMPLE:
        # M3b: final_norm reuses the dyn_acttail rmsnorm (k_rms). lm_head fp32-out
        # GEMM + streamed sampler.
        k_lmhead = Kernel(
            "llama_gemm_tiled_layer_K2048_N4_lmhead_fp32out",
            KO_GEMM,
            [t_D8_i8, t_wlm_slot, t_lm_chunk_f32, np.int32],
        )
        k_sample = Kernel(
            "llama_sample_streamed",
            KO_SAMP,
            [t_lm_chunk_f32, t_state_i8, t_params_u32, np.int32, np.int32, np.int32],
        )
        k_sample_final = Kernel(
            "llama_sample_streamed_finalize",
            KO_SAMP,
            [t_state_i8, _i32a(1), t_params_u32],
        )
    if GATHER:
        # embed select over a lm_head weight slot: pick embed[token] -> seed.
        k_embsel = Kernel(
            "llama_embed_select_slot",
            "llama_embed_select.cc.o",
            [t_wlm_slot, t_D8_i8, _i32a(1), np.int32],
        )
    # q_proj-mh fp32 out (2a): act_scale from h1 tail; q_out_scales downstream.
    k_q = Kernel(
        "llama_gemm_tiled_layer_K2048_N4_perchan_q_mh_fp32out_acttail",
        KO_GEMM2,
        [t_D8_i8, t_WQ_slot, _f32(QD), np.int32],
    )
    k_qrequant = Kernel("llama_q_requant", KO_GLUE, [_f32(QD), t_QF_i8])
    # o_proj-mh fp32 out, act_scale from the self-cal af tail (1b).
    k_o = Kernel(
        "llama_gemm_tiled_layer_K2048_N4_perchan_v2_o_mh_fp32out_acttail",
        KO_GEMM2,
        [t_AF8_i8, t_WO_slot, t_D_f32, np.int32],
    )
    # gate acttail (act_scale from h2 tail; inv_out stays silu-lock arg)
    k_gate = Kernel(
        "llama_gemm_tiled_layer_K2048_N4_perchan_gate_acttail",
        KO_GEMM,
        [t_D8_i8, t_WG_slot, t_HD_i8, np.int32, np.float32],
    )
    # up_proj fp32 out (2b): act_scale from h2 tail; up_out_scale downstream.
    k_up = Kernel(
        "llama_gemm_tiled_layer_K2048_N4_perchan_v2_up_fp32out_acttail",
        KO_GEMM,
        [t_D8_i8, t_WU_slot, _f32(HD), np.int32],
    )
    k_uprequant = Kernel("llama_up_requant", KO_SILU, [_f32(HD), t_UF_i8])
    k_down = Kernel(
        "llama_gemm_tiled_layer_K8192_N4_perchan_v2_d_fp32out_acttail",
        KO_GEMM,
        [t_UF_i8, t_WD_slot, t_D_f32, np.int32],
    )
    k_rope = Kernel("llama_rope_int8_mh_dyn", KO_ROPE, [t_QF_i8, t_CS_bf16, t_QR_i8])
    # k_proj / v_proj: fp32 out, act_scale from h1 tail (one symbol, both).
    k_kv = Kernel(
        "llama_gemm_tiled_layer_K2048_N4_perchan_fp32out_acttail",
        KO_GEMM2,
        [t_D8_i8, t_WK_slot, t_KVfp, np.int32],
    )
    k_qkvcomb = Kernel(
        "llama_qkv_combine",
        KO_GLUE,
        [t_QR_i8, t_KVCS_i8, t_COMB_HALF_i8, t_COMB_HALF_i8],
    )
    KO_KVA = "llama_kv_append.cc.o"
    # PERSIST uses the growing-cache append (slot = T_used, advances T_used ->
    # T_used+1 so position advances on-chip via the carried cache). Non-persist
    # keeps the original single-token append (slot = T_used-1, prefix unchanged).
    _append_sym = (
        "llama_kv_append_combined_grow" if PERSIST_GROW else "llama_kv_append_combined"
    )
    k_append = Kernel(_append_sym, KO_KVA, [t_COMB_i8, t_KV_i8, t_KV_i8])
    if PERSIST_RESIDENT:
        # Resident KV: per-head cache buffer holds N_LAYERS caches; the layer
        # index selects the slot. seed (token 0): host cache -> resident slot +
        # append. resident (tokens 1..): in-place append. flowkv: attend the
        # layer's resident slot.
        t_KVRES_i8 = _i8(N_LAYERS * PER_KV_HEAD_BYTES)
        k_append_seed = Kernel(
            "llama_kv_append_combined_resident_seed",
            KO_KVA,
            [t_COMB_i8, t_KV_i8, t_KVRES_i8, np.int32],
        )
        k_append_res = Kernel(
            "llama_kv_append_combined_resident",
            KO_KVA,
            [t_COMB_i8, t_KVRES_i8, np.int32],
        )
    # sv_merge_selfcal: de-interleave [body|sv_out_scale] chunks (1c).
    k_svmerge = Kernel(
        "llama_sv_merge_selfcal",
        KO_GLUE,
        [t_SVCONCAT_HALF_i8, t_SVCONCAT_HALF_i8, _i8(SVFULL_BYTES)],
    )
    # af_concat fully self-cal (1b+1c): 1 input (merged sv); o_act -> af tail.
    k_afconcat = Kernel(
        "llama_af_concat_selfcal2", KO_GLUE, [_i8(SVFULL_BYTES), t_AF8_i8]
    )
    # flowkv self-cal (1c): per-Q-head sv_out_scale -> out_chunk tail.
    k_fkv = Kernel(
        "llama_flowkv_mh_kvc_selfcal", KO_FKV, [t_COMB_i8, t_KV_i8, t_SVCHUNK_i8]
    )
    if PERSIST_RESIDENT:
        k_fkv_res = Kernel(
            "llama_flowkv_mh_kvc_selfcal_resident",
            KO_FKV,
            [t_COMB_i8, t_KVRES_i8, t_SVCHUNK_i8, np.int32],
        )
    # silu self-cal (1a): writes silu_out_scale to sf tail.
    k_silu = Kernel("llama_silu_mul_int8_selfcal", KO_SILU, [t_HD_i8, t_UF_i8, t_UF_i8])
    # rescale-add: (residual int8[D+8], proj fp32[D]) -> residual int8[D+8].
    KO_RADD = "llama_rescale_add.cc.o"
    k_add = Kernel("llama_rescale_add_D", KO_RADD, [t_D8_i8, t_D_f32, t_D8_i8])

    # --- Worker bodies (each loops N_LAYERS times) ---
    # Persistent loop: the 1:1 per-layer workers (rms/q/kv/attn/ffn/...) just see
    # PT*N_LAYERS activations stream by -- they're position-agnostic, so looping
    # CHAIN_ITERS handles all PT tokens. The token boundary lives only in the
    # router (seed source), add2 (residual reset), fnorm + onestream (per-token
    # sample), and the seedmux/feedback wiring. PT==1 => CHAIN_ITERS==N_LAYERS,
    # byte-identical to the non-persist chain.
    CHAIN_ITERS = PT * N_LAYERS
    if GATHER:
        # Front embed gather: stream the lm_head weight slots; select
        # embed[token_in] -> seed (int8[D]+scale = layer-0 input). One acquire of
        # the seed output + token; loop the LM_N_TILES weight slots.
        def w_gather(c_wlm, c_tok, c_seed, k):
            tok = c_tok.acquire(1)
            o = c_seed.acquire(1)
            for t in range_(LM_N_TILES):
                w = c_wlm.acquire(1)
                k(w, o, tok, _i32(t))
                c_wlm.release(1)
            c_seed.release(1)
            c_tok.release(1)

    def w_router(c_seed, c_back, p_routed):
        # Copy D+8: body + per-token residual scale tail (4 B scale + 4 B pad).
        # Persistent loop: repeat the (seed once, back N_LAYERS-1 times) pattern
        # for each of PT tokens. c_seed delivers the per-token layer-0 input
        # (token 0 = host xin, tokens 1..PT-1 = on-chip feedback via seedmux);
        # c_back is the same intra-token residual loop-back as before. PT==1 =>
        # identical to the non-persist router.
        for _ in range_(PT):
            x = c_seed.acquire(1)
            o = p_routed.acquire(1)
            for i in range_(D + 8):
                o[i] = x[i]
            p_routed.release(1)
            c_seed.release(1)
            for _l in range_(N_LAYERS - 1):
                x = c_back.acquire(1)
                o = p_routed.acquire(1)
                for i in range_(D + 8):
                    o[i] = x[i]
                p_routed.release(1)
                c_back.release(1)

    def w_rms(c_in, c_gamma, c_out, k):
        for _ in range_(CHAIN_ITERS):
            x = c_in.acquire(1)
            g = c_gamma.acquire(1)
            o = c_out.acquire(1)
            k(x, g, o)  # act_scale read from input tail x[D..D+4]
            c_in.release(1)
            c_gamma.release(1)
            c_out.release(1)

    def w_q(c_act, c_w, c_out, k):
        for _ in range_(CHAIN_ITERS):
            a = c_act.acquire(1)
            o = c_out.acquire(1)
            for t in range_(N_TILES_Q):
                w = c_w.acquire(1)
                k(a, w, o, _i32(t))
                c_w.release(1)
            c_act.release(1)
            c_out.release(1)

    # 1-in/1-out requant reduce stages (2a q, 2b up), per layer.
    def w_requant(c_in, c_out, k):
        for _ in range_(CHAIN_ITERS):
            x = c_in.acquire(1)
            o = c_out.acquire(1)
            k(x, o)
            c_in.release(1)
            c_out.release(1)

    def w_o(c_act, c_w, c_out, k):
        for _ in range_(CHAIN_ITERS):
            a = c_act.acquire(1)
            o = c_out.acquire(1)
            for t in range_(N_TILES_O):
                w = c_w.acquire(1)
                k(a, w, o, _i32(t))
                c_w.release(1)
            c_act.release(1)
            c_out.release(1)

    def w_gate(c_act, c_w, c_out, k):
        for _ in range_(CHAIN_ITERS):
            a = c_act.acquire(1)
            o = c_out.acquire(1)
            for t in range_(N_TILES_G):
                w = c_w.acquire(1)
                k(a, w, o, _i32(t), GATE_INV_OUT_SCALE)
                c_w.release(1)
            c_act.release(1)
            c_out.release(1)

    def w_up(c_act, c_w, c_out, k):
        for _ in range_(CHAIN_ITERS):
            a = c_act.acquire(1)
            o = c_out.acquire(1)
            for t in range_(N_TILES_U):
                w = c_w.acquire(1)
                k(a, w, o, _i32(t))
                c_w.release(1)
            c_act.release(1)
            c_out.release(1)

    def w_down(c_act, c_w, c_out, k):
        for _ in range_(CHAIN_ITERS):
            a = c_act.acquire(1)
            o = c_out.acquire(1)
            for t in range_(N_TILES_D):
                w = c_w.acquire(1)
                k(a, w, o, _i32(t))
                c_w.release(1)
            c_act.release(1)
            c_out.release(1)

    def w_rope(c_x, c_cs, c_out, k):
        for _ in range_(CHAIN_ITERS):
            x = c_x.acquire(1)
            cs = c_cs.acquire(1)
            o = c_out.acquire(1)
            k(x, cs, o)
            c_x.release(1)
            c_cs.release(1)
            c_out.release(1)

    # k_proj / v_proj: N_TILES_K iters -> fp32 KV_DIM.
    def w_kv(c_act, c_w, c_out, k):
        for _ in range_(CHAIN_ITERS):
            a = c_act.acquire(1)
            o = c_out.acquire(1)
            for t in range_(N_TILES_K):
                w = c_w.acquire(1)
                k(a, w, o, _i32(t))
                c_w.release(1)
            c_act.release(1)
            c_out.release(1)

    def w_qkvcomb(c_qr, c_kvcs, c_lo, c_hi, k):
        for _ in range_(CHAIN_ITERS):
            qr = c_qr.acquire(1)
            kvcs = c_kvcs.acquire(1)
            lo = c_lo.acquire(1)
            hi = c_hi.acquire(1)
            k(qr, kvcs, lo, hi)
            c_qr.release(1)
            c_kvcs.release(1)
            c_lo.release(1)
            c_hi.release(1)

    def w_svmerge(c_lo, c_hi, c_out, k):
        for _ in range_(CHAIN_ITERS):
            lo = c_lo.acquire(1)
            hi = c_hi.acquire(1)
            o = c_out.acquire(1)
            k(lo, hi, o)
            c_lo.release(1)
            c_hi.release(1)
            c_out.release(1)

    # af_concat (self-cal): 1 input (merged sv with bodies+sv_out_scales).
    def w_afconcat(c_in, c_out, k):
        for _ in range_(CHAIN_ITERS):
            x = c_in.acquire(1)
            o = c_out.acquire(1)
            k(x, o)
            c_in.release(1)
            c_out.release(1)

    # Co-located KV append + flowkv (per layer): append writes slot[pos] then
    # flowkv attends over the updated cache; cache_out drained device-owned.
    def w_attn(c_comb, c_kvin, c_kvout, c_sv, k_app, k_fk):
        for _ in range_(CHAIN_ITERS):
            comb = c_comb.acquire(1)
            kvin = c_kvin.acquire(1)
            kvout = c_kvout.acquire(1)
            sv = c_sv.acquire(1)
            k_app(comb, kvin, kvout)
            k_fk(comb, kvout, sv)
            c_comb.release(1)
            c_kvin.release(1)
            c_kvout.release(1)
            c_sv.release(1)

    # Resident-KV attn: this head's N_LAYERS caches live in b_kv (worker-local,
    # never DMA'd). token 0 seeds each layer's cache from the host (c_kvin, one
    # fill/layer) + appends; tokens 1..PT-1 append IN PLACE (no c_kvin). flowkv
    # attends the layer's resident slot. No kvout (cache never leaves the tile).
    def w_attn_resident(c_comb, c_kvin, c_sv, b_kv, k_seed, k_app_r, k_fk_r):
        # token 0: seed + attend, per layer.
        for L in range_(N_LAYERS):
            comb = c_comb.acquire(1)
            kvin = c_kvin.acquire(1)
            sv = c_sv.acquire(1)
            k_seed(comb, kvin, b_kv, _i32(L))
            k_fk_r(comb, b_kv, sv, _i32(L))
            c_comb.release(1)
            c_kvin.release(1)
            c_sv.release(1)
        # tokens 1..PT-1: in-place append + attend, per layer (no host KV).
        for _ in range_(PT - 1):
            for L in range_(N_LAYERS):
                comb = c_comb.acquire(1)
                sv = c_sv.acquire(1)
                k_app_r(comb, b_kv, _i32(L))
                k_fk_r(comb, b_kv, sv, _i32(L))
                c_comb.release(1)
                c_sv.release(1)

    def w_silu(c_g, c_u, c_out, k):
        for _ in range_(CHAIN_ITERS):
            g = c_g.acquire(1)
            u = c_u.acquire(1)
            o = c_out.acquire(1)
            k(g, u, o)
            c_g.release(1)
            c_u.release(1)
            c_out.release(1)

    # rescale-add: c_resid = residual int8[D+8], c_proj = projection fp32[D].
    def w_add1(c_resid, c_proj, c_out, k):
        for _ in range_(CHAIN_ITERS):
            r = c_resid.acquire(1)
            p = c_proj.acquire(1)
            o = c_out.acquire(1)
            k(r, p, o)
            c_resid.release(1)
            c_proj.release(1)
            c_out.release(1)

    # add2 with peeled final iter: writes to `back` for L=0..N-2, `out` for L=N-1.
    # Persistent loop: per token, the same peeled pattern -- back for the first
    # N_LAYERS-1 layers, out (-> fnorm -> onestream sample) for the last. PT==1 =>
    # identical to the non-persist add2.
    def w_add2(c_resid, c_proj, c_back, c_out, k):
        for _ in range_(PT):
            for _l in range_(N_LAYERS - 1):
                r = c_resid.acquire(1)
                p = c_proj.acquire(1)
                o = c_back.acquire(1)
                k(r, p, o)
                c_resid.release(1)
                c_proj.release(1)
                c_back.release(1)
            r = c_resid.acquire(1)
            p = c_proj.acquire(1)
            o = c_out.acquire(1)
            k(r, p, o)
            c_resid.release(1)
            c_proj.release(1)
            c_out.release(1)

    # --- M3b sample-block worker bodies (fire once, after the last layer) ---
    if SAMPLE:
        # final_norm: rms over the chain's final hidden (of_out int8[D+8]).
        # Persistent loop: fires once per token (PT times).
        def w_fnorm(c_in, c_gam, c_out, k):
            for _ in range_(PT):
                x = c_in.acquire(1)
                g = c_gam.acquire(1)
                o = c_out.acquire(1)
                k(x, g, o)
                c_in.release(1)
                c_gam.release(1)
                c_out.release(1)

        # lm_head GEMM: normed hidden x V int8 weights -> fp32 logit chunks into
        # the relay (which assembles the resident memtile buffer).
        def w_lmhead(c_act, c_w, rel, k):
            a = c_act.acquire(1)
            for _ch in range_(SAMPLE_N_CHUNKS):
                o = rel.gemm_acquire()
                for t in range_(SAMPLE_TILES_PER_CHUNK):
                    w = c_w.acquire(1)
                    k(a, w, o, _i32(t))
                    c_w.release(1)
                rel.gemm_release()
            c_act.release(1)

        # streamed sampler: SAMPLE_PASSES passes over the replayed logits.
        def w_sample(rel, st, p, c_tok, ks, kf):
            tok = c_tok.acquire(1)
            for pass_i in range_(1, SAMPLE_PASSES + 1):
                for ch in range_(SAMPLE_N_CHUNKS):
                    rb = rel.acquire(1)
                    ks(rb, st, p, _i32(pass_i), _i32(ch), _const_i32(0))
                    rel.release(1)
            kf(st, tok, p)
            c_tok.release(1)

        # One-stream: fused lm_head GEMM + top-k insert over the SINGLE 262 MB
        # table pass, then finalize -> token + next-token embed seed. No relay,
        # no replay, no second gather stream.
        def w_onestream(c_act, c_w, c_out, sl, sg, ss, sr, slen, pr, ki, kf):
            a = c_act.acquire(1)
            for t in range_(LM_N_TILES):
                w = c_w.acquire(1)
                ki(a, w, sl, sg, ss, sr, slen, _i32(t))
                c_w.release(1)
            c_act.release(1)
            o = c_out.acquire(1)
            kf(sl, sg, ss, sr, slen, pr, o)
            c_out.release(1)

        # Persistent loop variant: PT tokens, each emitting the packed output AND
        # feeding the next-token embed seed back ON-CHIP (-> seedmux -> router).
        # The packed buffer is [seed int8[D] | scale f32 | token i32 | pad]; the
        # feedback seed is the layer-0 input format [seed int8[D] | scale f32 |
        # pad], so copy o[:D+4] (body+scale) and zero the 4 B pad (the token bytes
        # in the packed buffer must NOT leak into the feedback's pad). Produces
        # feedback PT times; seedmux consumes PT-1 (the last token's seed is
        # produced but unused) -- the proven persist_decode imbalance, safe with
        # a depth>=2 feedback fifo.
        def w_onestream_persist(
            c_act, c_w, c_out, c_fb, sl, sg, ss, sr, slen, pr, ki, kf
        ):
            for _ in range_(PT):
                a = c_act.acquire(1)
                for t in range_(LM_N_TILES):
                    w = c_w.acquire(1)
                    ki(a, w, sl, sg, ss, sr, slen, _i32(t))
                    c_w.release(1)
                c_act.release(1)
                o = c_out.acquire(1)
                kf(sl, sg, ss, sr, slen, pr, o)
                fb = c_fb.acquire(1)
                for i in range_(D + 4):
                    fb[i] = o[i]
                for i in range_(4):
                    fb[D + 4 + i] = 0
                c_fb.release(1)
                c_out.release(1)

        # seedmux: token 0's layer-0 input from the host (of_hostseed); tokens
        # 1..PT-1 from the on-chip feedback. Keeps the router at 2 inputs.
        def w_seedmux(c_host, c_fb, p_seed):
            h = c_host.acquire(1)
            o = p_seed.acquire(1)
            for i in range_(D + 8):
                o[i] = h[i]
            p_seed.release(1)
            c_host.release(1)
            for _ in range_(PT - 1):
                f = c_fb.acquire(1)
                o = p_seed.acquire(1)
                for i in range_(D + 8):
                    o[i] = f[i]
                p_seed.release(1)
                c_fb.release(1)

    PSK = 8192
    ATSK = 16384
    # Resident-KV attn tiles hold the 2x17416 b_kv + 17416 seed-fill buffer, so
    # the stack must be small (flowkv ~1KB scratch). Tunable if placement fails.
    RESIDENT_ATSK = int(_os.environ.get("LLAMA_RESIDENT_ATSK", "8192"))
    # FFN-side stack: of_sf grows to HD+8 (silu self-cal) -> Bug-12 stack/fifo
    # alias at 8192 on the FFN tiles; 4096 clears it (see single-layer 1a).
    FFNSK = int(_os.environ.get("LLAMA_FFNSK", "4096"))

    workers = [
        # Router
        Worker(
            w_router,
            [of_seed.cons(), of_back.cons(), of_routed.prod()],
            tile=Tile(3, 4),
            stack_size=PSK,
        ),
        # Attention front
        Worker(
            w_rms,
            [of_routed.cons(), of_gam_in.cons(), of_h1.prod(), k_rms],
            tile=Tile(0, 2),
        ),
        Worker(
            w_q,
            [of_h1.cons(), of_wq.cons(), of_qfp.prod(), k_q],
            tile=Tile(0, 3),
            stack_size=PSK,
        ),
        # q_requant (2a): fp32 q -> per-head self-cal q_out_scale -> int8 + tail.
        Worker(w_requant, [of_qfp.cons(), of_qf.prod(), k_qrequant], tile=Tile(4, 3)),
        Worker(
            w_rope, [of_qf.cons(), of_cs.cons(), of_qr.prod(), k_rope], tile=Tile(0, 4)
        ),
        # k_proj / v_proj (h1 broadcast); fp32 out -> memtile kvcs join.
        Worker(
            w_kv,
            [of_h1.cons(), of_wk.cons(), kvcs_eps[0].prod(), k_kv],
            tile=Tile(7, 2),
            stack_size=PSK,
        ),
        Worker(
            w_kv,
            [of_h1.cons(), of_wv.cons(), kvcs_eps[1].prod(), k_kv],
            tile=Tile(7, 3),
            stack_size=PSK,
        ),
        Worker(
            w_qkvcomb,
            [
                of_qr.cons(),
                of_kvcs.cons(),
                of_comb_lo.prod(),
                of_comb_hi.prod(),
                k_qkvcomb,
            ],
            tile=Tile(0, 5),
        ),
        # 8 co-located append+flowkv workers (cols 1, 2 -- rows 2-5).
        # PERSIST_RESIDENT: KV cache resident in b_kv[i] (no kvout); host fills
        # of_kvs only on token 0 (seed). Otherwise: host-ferried kvin/kvout.
        *(
            [
                Worker(
                    w_attn_resident,
                    [
                        comb_eps[i].cons(),
                        of_kvs[i].cons(),
                        svchunk_eps[i].prod(),
                        b_kv[i],
                        k_append_seed,
                        k_append_res,
                        k_fkv_res,
                    ],
                    tile=Tile(1 + i // 4, 2 + i % 4),
                    # Resident KV: the 2x17416 b_kv buffer + the 17416 seed-fill
                    # cons buffer dominate the 64 KB tile, so the stack must be
                    # small (flowkv needs ~1 KB: scores[T]+qvals[T]; append
                    # k_rope[HD]). 8192 keeps total ~62 KB; ATSK (16384) overflows.
                    stack_size=RESIDENT_ATSK,
                )
                for i in range(N_HEADS_KV)
            ]
            if PERSIST_RESIDENT
            else [
                Worker(
                    w_attn,
                    [
                        comb_eps[i].cons(),
                        of_kvs[i].cons(),
                        kvout_eps[i].prod(),
                        svchunk_eps[i].prod(),
                        k_append,
                        k_fkv,
                    ],
                    tile=Tile(1 + i // 4, 2 + i % 4),
                    stack_size=ATSK,
                )
                for i in range(N_HEADS_KV)
            ]
        ),
        # Attention back
        Worker(
            w_svmerge,
            [of_svconcat_lo.cons(), of_svconcat_hi.cons(), of_svfull.prod(), k_svmerge],
            tile=Tile(3, 2),
        ),
        Worker(
            w_afconcat,
            [of_svfull.cons(), of_af.prod(), k_afconcat],
            tile=Tile(3, 3),
        ),
        Worker(
            w_o,
            [of_af.cons(), of_wo.cons(), of_op.prod(), k_o],
            tile=Tile(3, 5),
            stack_size=PSK,
        ),
        Worker(
            w_add1,
            [of_routed.cons(), of_op.cons(), of_x1.prod(), k_add],
            tile=Tile(6, 5),
        ),
        # FFN (FFNSK=4096: of_sf=HD+8 self-cal Bug-12 alias, see single layer)
        Worker(
            w_rms,
            [of_x1.cons(), of_gam_post.cons(), of_h2.prod(), k_rms],
            tile=Tile(6, 4),
        ),
        Worker(
            w_gate,
            [of_h2.cons(), of_wg.cons(), of_gf.prod(), k_gate],
            tile=Tile(4, 2),
            stack_size=FFNSK,
        ),
        Worker(
            w_up,
            [of_h2.cons(), of_wu.cons(), of_ufp.prod(), k_up],
            tile=Tile(5, 2),
            stack_size=FFNSK,
        ),
        # up_requant (2b): fp32 up -> self-cal up_out_scale -> int8 + tail.
        Worker(
            w_requant,
            [of_ufp.cons(), of_uf.prod(), k_uprequant],
            tile=Tile(5, 3),
            stack_size=FFNSK,
        ),
        Worker(
            w_silu,
            [of_gf.cons(), of_uf.cons(), of_sf.prod(), k_silu],
            tile=Tile(4, 5),
            stack_size=FFNSK,
        ),
        Worker(
            w_down,
            [of_sf.cons(), of_wd.cons(), of_df.prod(), k_down],
            tile=Tile(6, 2),
            stack_size=FFNSK,
        ),
        # add2 with peeled final iter
        Worker(
            w_add2,
            [of_x1.cons(), of_df.cons(), of_back.prod(), of_out.prod(), k_add],
            tile=Tile(7, 5),
        ),
    ]

    if SAMPLE and ONESTREAM:
        workers += [
            # final_norm (free tile 4,4)
            Worker(
                w_fnorm,
                [of_out.cons(), of_fgam.cons(), of_fnorm.prod(), k_rms],
                tile=Tile(4, 4),
            ),
            # Fused lm_head GEMM + top-k insert + finalize on one tile: ONE table
            # pass -> token + next-token embed seed. The resident top-k rows live
            # on this tile (KS*D bytes); KS<=8 fits the 64 KB tile alongside the
            # depth-2 wlm fifo at D=2048 (see onestream-topk-sampler memory).
            (
                Worker(
                    w_onestream_persist,
                    [
                        of_fnorm.cons(),
                        of_wlm.cons(),
                        of_opacked.prod(),
                        of_feedback.prod(),
                        b_set_logit,
                        b_set_gidx,
                        b_set_scale,
                        b_set_row,
                        b_set_len,
                        b_os_params,
                        k_os_insert,
                        k_os_final,
                    ],
                    tile=Tile(5, 4),
                    stack_size=ATSK,
                )
                if PERSIST
                else Worker(
                    w_onestream,
                    [
                        of_fnorm.cons(),
                        of_wlm.cons(),
                        of_opacked.prod(),
                        b_set_logit,
                        b_set_gidx,
                        b_set_scale,
                        b_set_row,
                        b_set_len,
                        b_os_params,
                        k_os_insert,
                        k_os_final,
                    ],
                    tile=Tile(5, 4),
                    stack_size=ATSK,
                )
            ),
        ]
        if PERSIST:
            # seedmux: host seed (token 0) vs on-chip feedback (tokens 1..PT-1)
            # -> of_seed (router input). Free tile (7,4).
            workers += [
                Worker(
                    w_seedmux,
                    [of_hostseed.cons(), of_feedback.cons(), of_seed.prod()],
                    tile=Tile(7, 4),
                    stack_size=PSK,
                ),
            ]
    elif SAMPLE:
        workers += [
            # final_norm (free tile 4,4)
            Worker(
                w_fnorm,
                [of_out.cons(), of_fgam.cons(), of_fnorm.prod(), k_rms],
                tile=Tile(4, 4),
            ),
            # lm_head GEMM (5,4 = relay gemm tile) + streamed sampler (5,5 =
            # relay sampler tile). The relay owns the memtile (4,1) DMA.
            Worker(
                w_lmhead,
                [of_fnorm.cons(), of_wlm.cons(), lm_relay, k_lmhead],
                tile=Tile(5, 4),
                stack_size=PSK,
            ),
            Worker(
                w_sample,
                [
                    lm_relay,
                    b_sample_state,
                    b_sample_params,
                    of_token.prod(),
                    k_sample,
                    k_sample_final,
                ],
                tile=Tile(5, 5),
                stack_size=8192,
            ),
        ]
    if GATHER:
        # Front gather produces of_seed (replaces the host xin fill). Free tile.
        workers += [
            Worker(
                w_gather,
                [of_gwlm.cons(), of_token_in.cons(), of_seed.prod(), k_embsel],
                tile=Tile(6, 3),
                stack_size=PSK,
            ),
        ]

    rt = Runtime()
    if ONESTREAM:
        # xin in, ONE packed output [seed int8[D] | scale f32 | token i32 | pad]
        # out -- 5 args (run_test segfaults at ~6). The persistent-loop shape:
        # seed feeds back as next layer-0 input, token rides along for the host.
        # PERSIST: the output holds PT packed records (one per generated token);
        # only token 0's xin comes from the host (rest feed back on-chip).
        rt_opacked_ty = _i8(PT * (D + 12))
        seq_ctx = rt.sequence(rt_xin_ty, rt_w_ty, rt_kv_ty, rt_lmw_ty, rt_opacked_ty)
    elif GATHER:
        # input is a token id (int32[1]); the device gathers embed[token] as the
        # layer-0 seed -> token->token dispatch.
        seq_ctx = rt.sequence(_i32a(1), rt_w_ty, rt_kv_ty, rt_lmw_ty, rt_token_ty)
    elif SAMPLE:
        seq_ctx = rt.sequence(rt_xin_ty, rt_w_ty, rt_kv_ty, rt_lmw_ty, rt_token_ty)
    else:
        seq_ctx = rt.sequence(rt_xin_ty, rt_w_ty, rt_kv_ty, rt_out_ty)
    with seq_ctx as _seq_args:
        opacked = None
        if ONESTREAM:
            xin, wblob, kvblob, lmw, opacked = _seq_args
            token = None
            out = None
        elif GATHER:
            token_in, wblob, kvblob, lmw, token = _seq_args
            xin = None
            out = None
        elif SAMPLE:
            xin, wblob, kvblob, lmw, token = _seq_args
            out = None
        else:
            xin, wblob, kvblob, out = _seq_args
        rt.start(*workers)

        tgs = []
        # Per-fifo per-layer fill specs.
        fill_specs = []

        def add_per_layer(
            prod, src, base_off, per_layer_stride, slot_bytes, slots_per_layer, total
        ):
            fill_specs.append(
                (
                    prod,
                    src,
                    base_off,
                    per_layer_stride,
                    slot_bytes,
                    slots_per_layer,
                    total,
                )
            )

        # Seed input (layer 0).
        seed_tg = rt.task_group()
        tgs.append(seed_tg)
        if GATHER:
            # The gather worker produces of_seed by selecting embed[token_in]
            # from a front pass over the lm_head weight slots. Fill the token id
            # + stream the embed weights (front, before the layer weights so the
            # gather completes before layer 0 consumes the seed).
            rt.fill(of_token_in.prod(), token_in, task_group=seed_tg)
            gw_tgs = [rt.task_group() for _ in range(LM_N_TILES)]
            for t in range(LM_N_TILES):
                rt.fill(
                    of_gwlm.prod(),
                    lmw,
                    tap=strided_tap(
                        LMW_GAMMA_BYTES + WLM_TOTAL,
                        LMW_GAMMA_BYTES + t * WLM_SLOT,
                        WLM_SLOT,
                        WLM_SLOT,
                        1,
                    ),
                    task_group=gw_tgs[t],
                    wait=True,
                )
                if t >= 2:
                    rt.finish_task_group(gw_tgs[t - 2])
            for t in range(max(0, LM_N_TILES - 2), LM_N_TILES):
                rt.finish_task_group(gw_tgs[t])
        elif PERSIST:
            # Persistent loop: only token 0's layer-0 input comes from the host
            # (-> of_hostseed -> seedmux); tokens 1..PT-1 feed back on-chip.
            rt.fill(of_hostseed.prod(), xin, task_group=seed_tg)
        else:
            rt.fill(of_seed.prod(), xin, task_group=seed_tg)

        # Weights (all from wblob).
        add_per_layer(
            of_gam_in.prod(),
            wblob,
            OFF_GAMMA_IN,
            GAMMA_BYTES,
            GAMMA_BYTES,
            1,
            WEIGHTS_BYTES,
        )
        add_per_layer(
            of_wq.prod(),
            wblob,
            OFF_WQ,
            N_TILES_Q * WQ_SLOT,
            WQ_SLOT,
            N_TILES_Q,
            WEIGHTS_BYTES,
        )
        # cos/sin for q-rope. Non-grow: one pair per layer (OFF_CS, same every
        # token). PERSIST_GROW issues these per-token in issue_token() from the
        # per-position block, so skip the static fill here.
        if not PERSIST_GROW:
            add_per_layer(
                of_cs.prod(), wblob, OFF_CS, CS_BYTES, CS_BYTES, 1, WEIGHTS_BYTES
            )
        add_per_layer(
            of_wo.prod(),
            wblob,
            OFF_WO,
            N_TILES_O * WO_SLOT,
            WO_SLOT,
            N_TILES_O,
            WEIGHTS_BYTES,
        )
        # (of_afscales fill removed -- sv_out_scales/o_act self-calibrated.)
        add_per_layer(
            of_gam_post.prod(),
            wblob,
            OFF_GAMMA_POST,
            GAMMA_BYTES,
            GAMMA_BYTES,
            1,
            WEIGHTS_BYTES,
        )
        add_per_layer(
            of_wg.prod(),
            wblob,
            OFF_WG,
            N_TILES_G * WG_SLOT,
            WG_SLOT,
            N_TILES_G,
            WEIGHTS_BYTES,
        )
        add_per_layer(
            of_wu.prod(),
            wblob,
            OFF_WU,
            N_TILES_U * WU_SLOT,
            WU_SLOT,
            N_TILES_U,
            WEIGHTS_BYTES,
        )
        add_per_layer(
            of_wd.prod(),
            wblob,
            OFF_WD,
            N_TILES_D * WD_SLOT,
            WD_SLOT,
            N_TILES_D,
            WEIGHTS_BYTES,
        )
        add_per_layer(
            of_wk.prod(),
            wblob,
            OFF_WK,
            N_TILES_K * WK_SLOT,
            WK_SLOT,
            N_TILES_K,
            WEIGHTS_BYTES,
        )
        add_per_layer(
            of_wv.prod(),
            wblob,
            OFF_WV,
            N_TILES_V * WV_SLOT,
            WV_SLOT,
            N_TILES_V,
            WEIGHTS_BYTES,
        )
        # cos/sin for append's rope_k (same host cos/sin as rope, OFF_CS).
        # PERSIST_GROW issues this per-token from the per-position block.
        if not PERSIST_GROW:
            add_per_layer(
                kvcs_eps[2].prod(),
                wblob,
                OFF_CS,
                CS_BYTES,
                CS_PACK_BYTES,
                1,
                WEIGHTS_BYTES,
            )

        # 2 KV fills per layer (lo / hi halves; each fans out to 4 attn workers).
        kv_specs = [
            (of_kv_lo.prod(), kvblob, 0, PER_LAYER_KV, KV_HALF_BYTES, 1, KV_BYTES),
            (
                of_kv_hi.prod(),
                kvblob,
                KV_HALF_BYTES,
                PER_LAYER_KV,
                KV_HALF_BYTES,
                1,
                KV_BYTES,
            ),
        ]

        # Device-owned cache: drain each layer's updated cache back over kvblob
        # (per-layer, lo/hi halves). Mirrors the kv_specs fills as drains.
        # PERSIST increment 1 holds KV at a fixed position: fills come from the
        # pristine region (offset 0) every token, drains go to a SCRATCH region
        # (offset KV_BYTES, ignored) so each token sees identical starting KV.
        # PERSIST increment 2 (GROWING KV): ping-pong two KV regions so token t's
        # drained (grown) cache becomes token t+1's fill. token t fills region
        # t%2, drains region (t+1)%2; token 0 fills region 0 (host pristine, with
        # T_used=P). The growing-append advances T_used on-chip each token, so the
        # carried cache accumulates context. kv_*_base computed per token below.
        # PERSIST_RESIDENT: 1x KV buffer, host fills only the token-0 seed, no
        # drain (cache resident in b_kv).
        kv_total = (2 * KV_BYTES) if (PERSIST and not PERSIST_RESIDENT) else KV_BYTES
        kvout_specs = (
            []
            if PERSIST_RESIDENT
            else [
                (of_kvout_lo.cons(), 0),
                (of_kvout_hi.cons(), KV_HALF_BYTES),
            ]
        )

        PINGPONG_DEPTH = int(_os.environ.get("LLAMA_CHAIN_PINGPONG", "2"))

        # One token's fills + drains. tok indexes the packed-output record. For
        # PT==1 (non-persist) this runs once and is byte-identical to before.
        def issue_token(tok):
            # Drains BEFORE fills to avoid the depth=1 drain-blocked deadlock.
            out_tg = rt.task_group()
            tgs.append(out_tg)
            if SAMPLE:
                # final_norm gamma from the lmw prefix (per token). The 32064
                # lm_head weight fills are issued AFTER the chain per-layer fills
                # so the chain runs first and produces of_out.
                rt.fill(
                    of_fgam.prod(),
                    lmw,
                    tap=strided_tap(
                        LMW_GAMMA_BYTES + WLM_TOTAL,
                        0,
                        LMW_GAMMA_BYTES,
                        LMW_GAMMA_BYTES,
                        1,
                    ),
                    task_group=out_tg,
                )
                if ONESTREAM:
                    # packed output [seed | scale | token | pad]; PERSIST writes
                    # PT records, one per token at offset tok*(D+12).
                    if PERSIST:
                        rt.drain(
                            of_opacked.cons(),
                            opacked,
                            tap=strided_tap(
                                PT * (D + 12), tok * (D + 12), D + 12, D + 12, 1
                            ),
                            wait=True,
                            task_group=out_tg,
                        )
                    else:
                        rt.drain(
                            of_opacked.cons(), opacked, wait=True, task_group=out_tg
                        )
                else:
                    rt.drain(of_token.cons(), token, wait=True, task_group=out_tg)
            else:
                rt.drain(of_out.cons(), out, wait=True, task_group=out_tg)

            # KV region selection:
            #  - PERSIST_GROW: ping-pong. token t fills region t%2, drains region
            #    (t+1)%2 -> the grown cache carries forward (real autoregressive).
            #  - PERSIST (fixed-pos, increment 1): always fill region 0
            #    (pristine), drain region 1 (scratch) -> every token sees the same
            #    KV.
            #  - non-persist: single region.
            if PERSIST_GROW:
                kv_fill_base = (tok % 2) * KV_BYTES
                kv_drain_base = ((tok + 1) % 2) * KV_BYTES
            elif PERSIST:
                kv_fill_base = 0
                kv_drain_base = KV_BYTES
            else:
                kv_fill_base = 0
                kv_drain_base = 0

            # Ping-pong per-layer fill issue with BD reuse.
            layer_tgs = [rt.task_group() for _ in range(N_LAYERS)]
            for L in range(N_LAYERS):
                for cons, half_off in kvout_specs:
                    rt.drain(
                        cons,
                        kvblob,
                        tap=strided_tap(
                            kv_total,
                            kv_drain_base + half_off + L * PER_LAYER_KV,
                            KV_HALF_BYTES,
                            KV_HALF_BYTES,
                            1,
                        ),
                        task_group=layer_tgs[L],
                        wait=True,
                    )
                for (
                    prod,
                    src,
                    base_off,
                    per_layer_stride,
                    slot_bytes,
                    slots_per_layer,
                    total,
                ) in fill_specs:
                    rt.fill(
                        prod,
                        src,
                        tap=strided_tap(
                            total,
                            base_off + L * per_layer_stride,
                            slot_bytes,
                            slot_bytes,
                            slots_per_layer,
                        ),
                        task_group=layer_tgs[L],
                        wait=True,
                    )
                # KV fills. PERSIST_RESIDENT: only token 0 seeds the resident
                # cache (the attn worker consumes of_kvs only on token 0); later
                # tokens read/write b_kv in place -> no KV DMA. Otherwise: every
                # token fills (parity-selected region + 2x total).
                if (not PERSIST_RESIDENT) or tok == 0:
                    for (
                        prod,
                        src,
                        base_off,
                        per_layer_stride,
                        slot_bytes,
                        slots_per_layer,
                        total,
                    ) in kv_specs:
                        rt.fill(
                            prod,
                            src,
                            tap=strided_tap(
                                kv_total if PERSIST else total,
                                kv_fill_base + base_off + L * per_layer_stride,
                                slot_bytes,
                                slot_bytes,
                                slots_per_layer,
                            ),
                            task_group=layer_tgs[L],
                            wait=True,
                        )
                if PERSIST_GROW:
                    # Per-position cos/sin: token `tok` at position P0+tok uses
                    # its own (cos,sin) for both q-rope (of_cs) and append rope_k
                    # (kvcs_eps[2]). Read from the per-position block at
                    # OFF_CS_PERPOS + (tok*N_LAYERS + L)*CS_BYTES.
                    cs_off = OFF_CS_PERPOS + (tok * N_LAYERS + L) * CS_BYTES
                    rt.fill(
                        of_cs.prod(),
                        wblob,
                        tap=strided_tap(WEIGHTS_BYTES, cs_off, CS_BYTES, CS_BYTES, 1),
                        task_group=layer_tgs[L],
                        wait=True,
                    )
                    rt.fill(
                        kvcs_eps[2].prod(),
                        wblob,
                        tap=strided_tap(
                            WEIGHTS_BYTES, cs_off, CS_PACK_BYTES, CS_PACK_BYTES, 1
                        ),
                        task_group=layer_tgs[L],
                        wait=True,
                    )
                if L >= PINGPONG_DEPTH:
                    rt.finish_task_group(layer_tgs[L - PINGPONG_DEPTH])
            for L in range(max(0, N_LAYERS - PINGPONG_DEPTH), N_LAYERS):
                rt.finish_task_group(layer_tgs[L])

            if SAMPLE:
                # lm_head weight stream (262 MB), issued AFTER the chain layer
                # fills so the chain runs first and produces of_out.
                lm_tgs = [rt.task_group() for _ in range(LM_N_TILES)]
                for t in range(LM_N_TILES):
                    rt.fill(
                        of_wlm.prod(),
                        lmw,
                        tap=strided_tap(
                            LMW_GAMMA_BYTES + WLM_TOTAL,
                            LMW_GAMMA_BYTES + t * WLM_SLOT,
                            WLM_SLOT,
                            WLM_SLOT,
                            1,
                        ),
                        task_group=lm_tgs[t],
                        wait=True,
                    )
                    if t >= 2:
                        rt.finish_task_group(lm_tgs[t - 2])
                for t in range(max(0, LM_N_TILES - 2), LM_N_TILES):
                    rt.finish_task_group(lm_tgs[t])

        for tok in range(PT):
            issue_token(tok)

        for tg in tgs:
            rt.finish_task_group(tg)

    return Program(NPU2(), rt).resolve_program()


def main():
    argparse.ArgumentParser().parse_args(sys.argv[1:])
    print(build())


if __name__ == "__main__":
    main()
