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

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern
from aie.dialects.arith import index_cast
from aie.ir import IntegerType


def _i32(idx):
    return index_cast(IntegerType.get_signless(32), idx)


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

SVCHUNK_BYTES = REP * HEAD_D  # 256
SV_CONCAT_HALF = (N_HEADS_KV // 2) * SVCHUNK_BYTES  # 1024
AF_BYTES = N_HEADS_Q * HEAD_D  # 2048

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
    rt_kv_ty = _i8(KV_BYTES)
    rt_out_ty = _i8(D + 8)

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

    of_gam_in = ObjectFifo(t_D_bf16, depth=1, name="gam_in")
    of_h1 = ObjectFifo(t_D8_i8, depth=1, name="h1")
    of_wq = ObjectFifo(t_WQ_slot, depth=2, name="wq")
    of_wk = ObjectFifo(t_WK_slot, depth=2, name="wk")
    of_wv = ObjectFifo(t_WV_slot, depth=2, name="wv")
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

    of_svfull = ObjectFifo(_i8(AF_BYTES), depth=1, name="sv_full")
    of_afscales = ObjectFifo(t_AFSCALES_i8, depth=1, name="af_scales")
    of_af = ObjectFifo(t_AF_i8, depth=1, name="af")
    of_wo = ObjectFifo(t_WO_slot, depth=2, name="wo")
    of_op = ObjectFifo(t_D_f32, depth=1, name="op")  # fp32 o_proj output

    of_x1 = ObjectFifo(t_D8_i8, depth=1, name="x1")  # residual int8[D+8]

    of_gam_post = ObjectFifo(t_D_bf16, depth=1, name="gam_post")
    of_h2 = ObjectFifo(t_D8_i8, depth=1, name="h2")
    of_wg = ObjectFifo(t_WG_slot, depth=2, name="wg")
    of_wu = ObjectFifo(t_WU_slot, depth=2, name="wu")
    of_wd = ObjectFifo(t_WD_slot, depth=1, name="wd")  # K=8192 L1 budget
    of_gf = ObjectFifo(t_HD_i8, depth=1, name="gf")
    of_uf = ObjectFifo(t_UF_i8, depth=1, name="uf")
    of_sf = ObjectFifo(t_HD_i8, depth=1, name="sf")
    of_df = ObjectFifo(t_D_f32, depth=1, name="df")  # fp32 down output

    KO_RMS = "llama_rmsnorm_int8.cc.o"
    KO_GEMM = "llama_gemm_int8_srs_tiled_layer.cc.o"
    KO_GEMM2 = "llama_gemm_int8_srs_tiled_layer_mh.cc.o"
    KO_ROPE = "llama_rope_int8_mh.cc.o"
    KO_SILU = "llama_silu_mul_int8.cc.o"
    KO_FKV = "llama_flowkv_mh.cc.o"
    KO_GLUE = "llama_gqa_glue.cc.o"
    KO_PT = "llama_layer_pt.cc.o"

    # rmsnorm acttail: per-token input scale read from the residual tail.
    k_rms = Kernel(
        "llama_rmsnorm_int8_dyn_acttail", KO_RMS, [t_D8_i8, t_D_bf16, t_D8_i8]
    )
    k_q = Kernel(
        "llama_gemm_tiled_layer_K2048_N4_perchan_v2_up_q_mh_acttail",
        KO_GEMM2,
        [t_D8_i8, t_WQ_slot, t_QF_i8, np.int32],
    )
    # o_proj-mh, fp32 output (consumed by rescale-add)
    k_o = Kernel(
        "llama_gemm_tiled_layer_K2048_N4_perchan_v2_o_mh_fp32out",
        KO_GEMM2,
        [t_AF_i8, t_WO_slot, t_D_f32, np.int32],
    )
    # gate acttail (act_scale from h2 tail; inv_out stays silu-lock arg)
    k_gate = Kernel(
        "llama_gemm_tiled_layer_K2048_N4_perchan_gate_acttail",
        KO_GEMM,
        [t_D8_i8, t_WG_slot, t_HD_i8, np.int32, np.float32],
    )
    k_up = Kernel(
        "llama_gemm_tiled_layer_K2048_N4_perchan_v2_up_u_acttail",
        KO_GEMM,
        [t_D8_i8, t_WU_slot, t_UF_i8, np.int32],
    )
    k_down = Kernel(
        "llama_gemm_tiled_layer_K8192_N4_perchan_v2_d_fp32out",
        KO_GEMM,
        [t_HD_i8, t_WD_slot, t_D_f32, np.int32],
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
    k_append = Kernel("llama_kv_append_combined", KO_KVA, [t_COMB_i8, t_KV_i8, t_KV_i8])
    k_svmerge = Kernel(
        "llama_sv_merge",
        KO_GLUE,
        [t_SVCONCAT_HALF_i8, t_SVCONCAT_HALF_i8, _i8(AF_BYTES)],
    )
    k_afconcat = Kernel(
        "llama_af_concat", KO_GLUE, [_i8(AF_BYTES), t_AFSCALES_i8, t_AF_i8]
    )
    # flowkv consumes the combined chunk (reads first 288 B q_chunk); kv input
    # is the post-append cache.
    k_fkv = Kernel("llama_flowkv_mh_kvc", KO_FKV, [t_COMB_i8, t_KV_i8, t_SVCHUNK_i8])
    k_silu = Kernel("llama_silu_mul_int8_dyn", KO_SILU, [t_HD_i8, t_UF_i8, t_HD_i8])
    # rescale-add: (residual int8[D+8], proj fp32[D]) -> residual int8[D+8].
    KO_RADD = "llama_rescale_add.cc.o"
    k_add = Kernel("llama_rescale_add_D", KO_RADD, [t_D8_i8, t_D_f32, t_D8_i8])

    # --- Worker bodies (each loops N_LAYERS times) ---
    def w_router(c_seed, c_back, p_routed):
        # Copy D+8: body + per-token residual scale tail (4 B scale + 4 B pad).
        x = c_seed.acquire(1)
        o = p_routed.acquire(1)
        for i in range_(D + 8):
            o[i] = x[i]
        p_routed.release(1)
        c_seed.release(1)
        for _ in range_(N_LAYERS - 1):
            x = c_back.acquire(1)
            o = p_routed.acquire(1)
            for i in range_(D + 8):
                o[i] = x[i]
            p_routed.release(1)
            c_back.release(1)

    def w_rms(c_in, c_gamma, c_out, k):
        for _ in range_(N_LAYERS):
            x = c_in.acquire(1)
            g = c_gamma.acquire(1)
            o = c_out.acquire(1)
            k(x, g, o)  # act_scale read from input tail x[D..D+4]
            c_in.release(1)
            c_gamma.release(1)
            c_out.release(1)

    def w_q(c_act, c_w, c_out, k):
        for _ in range_(N_LAYERS):
            a = c_act.acquire(1)
            o = c_out.acquire(1)
            for t in range_(N_TILES_Q):
                w = c_w.acquire(1)
                k(a, w, o, _i32(t))
                c_w.release(1)
            c_act.release(1)
            c_out.release(1)

    def w_o(c_act, c_w, c_out, k):
        for _ in range_(N_LAYERS):
            a = c_act.acquire(1)
            o = c_out.acquire(1)
            for t in range_(N_TILES_O):
                w = c_w.acquire(1)
                k(a, w, o, _i32(t))
                c_w.release(1)
            c_act.release(1)
            c_out.release(1)

    def w_gate(c_act, c_w, c_out, k):
        for _ in range_(N_LAYERS):
            a = c_act.acquire(1)
            o = c_out.acquire(1)
            for t in range_(N_TILES_G):
                w = c_w.acquire(1)
                k(a, w, o, _i32(t), GATE_INV_OUT_SCALE)
                c_w.release(1)
            c_act.release(1)
            c_out.release(1)

    def w_up(c_act, c_w, c_out, k):
        for _ in range_(N_LAYERS):
            a = c_act.acquire(1)
            o = c_out.acquire(1)
            for t in range_(N_TILES_U):
                w = c_w.acquire(1)
                k(a, w, o, _i32(t))
                c_w.release(1)
            c_act.release(1)
            c_out.release(1)

    def w_down(c_act, c_w, c_out, k):
        for _ in range_(N_LAYERS):
            a = c_act.acquire(1)
            o = c_out.acquire(1)
            for t in range_(N_TILES_D):
                w = c_w.acquire(1)
                k(a, w, o, _i32(t))
                c_w.release(1)
            c_act.release(1)
            c_out.release(1)

    def w_rope(c_x, c_cs, c_out, k):
        for _ in range_(N_LAYERS):
            x = c_x.acquire(1)
            cs = c_cs.acquire(1)
            o = c_out.acquire(1)
            k(x, cs, o)
            c_x.release(1)
            c_cs.release(1)
            c_out.release(1)

    # k_proj / v_proj: N_TILES_K iters -> fp32 KV_DIM.
    def w_kv(c_act, c_w, c_out, k):
        for _ in range_(N_LAYERS):
            a = c_act.acquire(1)
            o = c_out.acquire(1)
            for t in range_(N_TILES_K):
                w = c_w.acquire(1)
                k(a, w, o, _i32(t))
                c_w.release(1)
            c_act.release(1)
            c_out.release(1)

    def w_qkvcomb(c_qr, c_kvcs, c_lo, c_hi, k):
        for _ in range_(N_LAYERS):
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
        for _ in range_(N_LAYERS):
            lo = c_lo.acquire(1)
            hi = c_hi.acquire(1)
            o = c_out.acquire(1)
            k(lo, hi, o)
            c_lo.release(1)
            c_hi.release(1)
            c_out.release(1)

    def w_afconcat(c_in, c_sc, c_out, k):
        for _ in range_(N_LAYERS):
            x = c_in.acquire(1)
            s = c_sc.acquire(1)
            o = c_out.acquire(1)
            k(x, s, o)
            c_in.release(1)
            c_sc.release(1)
            c_out.release(1)

    # Co-located KV append + flowkv (per layer): append writes slot[pos] then
    # flowkv attends over the updated cache; cache_out drained device-owned.
    def w_attn(c_comb, c_kvin, c_kvout, c_sv, k_app, k_fk):
        for _ in range_(N_LAYERS):
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

    def w_silu(c_g, c_u, c_out, k):
        for _ in range_(N_LAYERS):
            g = c_g.acquire(1)
            u = c_u.acquire(1)
            o = c_out.acquire(1)
            k(g, u, o)
            c_g.release(1)
            c_u.release(1)
            c_out.release(1)

    # rescale-add: c_resid = residual int8[D+8], c_proj = projection fp32[D].
    def w_add1(c_resid, c_proj, c_out, k):
        for _ in range_(N_LAYERS):
            r = c_resid.acquire(1)
            p = c_proj.acquire(1)
            o = c_out.acquire(1)
            k(r, p, o)
            c_resid.release(1)
            c_proj.release(1)
            c_out.release(1)

    # add2 with peeled final iter: writes to `back` for L=0..N-2, `out` for L=N-1.
    def w_add2(c_resid, c_proj, c_back, c_out, k):
        for _ in range_(N_LAYERS - 1):
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

    PSK = 8192
    ATSK = 16384

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
            [of_h1.cons(), of_wq.cons(), of_qf.prod(), k_q],
            tile=Tile(0, 3),
            stack_size=PSK,
        ),
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
        # 8 co-located append+flowkv workers (cols 1, 2 -- rows 2-5)
        *[
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
        ],
        # Attention back
        Worker(
            w_svmerge,
            [of_svconcat_lo.cons(), of_svconcat_hi.cons(), of_svfull.prod(), k_svmerge],
            tile=Tile(3, 2),
        ),
        Worker(
            w_afconcat,
            [of_svfull.cons(), of_afscales.cons(), of_af.prod(), k_afconcat],
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
        # FFN (match layer_d2048 placement; Bug 12: PSK=8192 only)
        Worker(
            w_rms,
            [of_x1.cons(), of_gam_post.cons(), of_h2.prod(), k_rms],
            tile=Tile(6, 4),
        ),
        Worker(
            w_gate,
            [of_h2.cons(), of_wg.cons(), of_gf.prod(), k_gate],
            tile=Tile(4, 2),
            stack_size=PSK,
        ),
        Worker(
            w_up,
            [of_h2.cons(), of_wu.cons(), of_uf.prod(), k_up],
            tile=Tile(5, 2),
            stack_size=PSK,
        ),
        Worker(
            w_silu,
            [of_gf.cons(), of_uf.cons(), of_sf.prod(), k_silu],
            tile=Tile(4, 5),
            stack_size=PSK,
        ),
        Worker(
            w_down,
            [of_sf.cons(), of_wd.cons(), of_df.prod(), k_down],
            tile=Tile(6, 2),
            stack_size=PSK,
        ),
        # add2 with peeled final iter
        Worker(
            w_add2,
            [of_x1.cons(), of_df.cons(), of_back.prod(), of_out.prod(), k_add],
            tile=Tile(7, 5),
        ),
    ]

    rt = Runtime()
    with rt.sequence(rt_xin_ty, rt_w_ty, rt_kv_ty, rt_out_ty) as (
        xin,
        wblob,
        kvblob,
        out,
    ):
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
        add_per_layer(of_cs.prod(), wblob, OFF_CS, CS_BYTES, CS_BYTES, 1, WEIGHTS_BYTES)
        add_per_layer(
            of_wo.prod(),
            wblob,
            OFF_WO,
            N_TILES_O * WO_SLOT,
            WO_SLOT,
            N_TILES_O,
            WEIGHTS_BYTES,
        )
        add_per_layer(
            of_afscales.prod(),
            wblob,
            OFF_AF_SCALES,
            AF_SCALES_BYTES,
            AF_SCALES_BYTES,
            1,
            WEIGHTS_BYTES,
        )
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
        add_per_layer(
            kvcs_eps[2].prod(), wblob, OFF_CS, CS_BYTES, CS_PACK_BYTES, 1, WEIGHTS_BYTES
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

        # Drains BEFORE fills to avoid the depth=1 drain-blocked deadlock.
        out_tg = rt.task_group()
        tgs.append(out_tg)
        rt.drain(of_out.cons(), out, wait=True, task_group=out_tg)

        # Device-owned cache: drain each layer's updated cache back over kvblob
        # (per-layer, lo/hi halves). Mirrors the kv_specs fills as drains.
        kvout_specs = [
            (of_kvout_lo.cons(), 0),
            (of_kvout_hi.cons(), KV_HALF_BYTES),
        ]

        # Ping-pong per-layer fill issue with BD reuse.
        PINGPONG_DEPTH = int(_os.environ.get("LLAMA_CHAIN_PINGPONG", "2"))
        layer_tgs = [rt.task_group() for _ in range(N_LAYERS)]
        for L in range(N_LAYERS):
            # Per-layer cache drains (issued in the same task group as fills).
            for cons, half_off in kvout_specs:
                rt.drain(
                    cons,
                    kvblob,
                    tap=strided_tap(
                        KV_BYTES,
                        half_off + L * PER_LAYER_KV,
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
            ) in (
                fill_specs + kv_specs
            ):
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
            if L >= PINGPONG_DEPTH:
                rt.finish_task_group(layer_tgs[L - PINGPONG_DEPTH])
        for L in range(max(0, N_LAYERS - PINGPONG_DEPTH), N_LAYERS):
            rt.finish_task_group(layer_tgs[L])

        for tg in tgs:
            rt.finish_task_group(tg)

    return Program(NPU2(), rt).resolve_program()


def main():
    argparse.ArgumentParser().parse_args(sys.argv[1:])
    print(build())


if __name__ == "__main__":
    main()
