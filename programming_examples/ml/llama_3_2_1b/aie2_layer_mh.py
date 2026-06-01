"""Phase 7a: full single-layer Llama 3.2 1B decoder at PRODUCTION
multi-head shapes (N_HEADS_Q=32, N_HEADS_KV=8, REP=4, HEAD_DIM=64,
Q_DIM=2048, D=2048, HD=8192, T=128). One xclbin, one dispatch, all
attention + FFN.

Topology (additions vs aie2_layer_d2048):
  - q_proj writes QD=2048 (32 heads) plus a 256 B per-head scale tail
    (32 [q_out_scale, sv_inv_out_scale] pairs).
  - rope_mh loops 32 heads, passes tail through.
  - q_split reorders rope's flat (QD body || 256 B tail) into 8 KV-group
    chunks of [REP*HEAD_DIM body || REP*8 B scale tail] = 288 B each.
  - memtile fanout: 8 ObjectFifos via .cons().split().
  - 8 attn workers (one per KV head, fused qk+sv via llama_flowkv_mh).
  - memtile concat: 8 ObjectFifos via .prod().join() into one 2048 B
    sv_concat buffer.
  - af_concat applies per-Q-head sv_out_scale + global o_inv_act_scale
    requant -> af (i8, 2048 B).
  - o_proj is now K=QD=2048 (vs K=64 single-head).
  - FFN half is unchanged.

KV cache: host-pre-filled, 8 ObjectFifos (one per KV head). Each kcache/
vcache fifo carries a 4 B scale header + 8192 B body. flowkv_mh reads
those at fixed offsets.

All ObjectFifos depth=1 (Bug 8). GEMM workers get stack_size=16384
(Bug 5). PSK applied to all large gemm workers.
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
D            = 2048
HD           = 8192
HEAD_D       = 64
N_HEADS_Q    = 32
N_HEADS_KV   = 8
REP          = N_HEADS_Q // N_HEADS_KV     # 4
QD           = N_HEADS_Q * HEAD_D          # 2048
KVD_PER_HEAD = HEAD_D                      # 64
T            = int(_os.environ.get("LLAMA_LAYER_T", "128"))
N_TILE       = 4

# Per-Q-head tail: 32 * [q_out_scale fp32, sv_inv_out_scale fp32] = 256 B.
QF_TAIL      = N_HEADS_Q * 8
QF_BYTES     = QD + QF_TAIL                # 2304
QR_BYTES     = QF_BYTES                    # rope passes tail through
# q-split output: 8 chunks of [REP*HEAD_D body || REP*8 scale tail].
QCHUNK_BODY  = REP * HEAD_D                # 256
QCHUNK_TAIL  = REP * 8                     # 32
QCHUNK_BYTES = QCHUNK_BODY + QCHUNK_TAIL   # 288
QCHUNKS_HALF_BYTES = (N_HEADS_KV // 2) * QCHUNK_BYTES  # 1152 (4 chunks)

# attn worker output: REP heads of HEAD_D each.
SVCHUNK_BYTES = REP * HEAD_D               # 256
SV_CONCAT_HALF = (N_HEADS_KV // 2) * SVCHUNK_BYTES  # 1024 (4 chunks)
AF_BYTES       = N_HEADS_Q * HEAD_D                  # 2048

# kv-per-head: 4 B scale header + T*HEAD_D body.
KV_HEADER     = 4
KCACHE_BYTES  = T * HEAD_D                 # 8192
VCACHE_BYTES  = T * HEAD_D                 # 8192
KCACHE_PADDED = KV_HEADER + KCACHE_BYTES   # 8196
VCACHE_PADDED = KV_HEADER + VCACHE_BYTES   # 8196

# af_concat scales buffer (host-packed): 32 sv_out_scales + o_inv_act_scale + pad.
AF_SCALES_BYTES = 192

# Slot prefixes (Bug 7: must be multiple of 64 B for weight body alignment).
PREFIX_ALIGN = 64
WQ_PREFIX = 448   # mh q_proj: act_scale + 32 q_inv_outs + 32 [q_out_scale, sv_inv_out_scale] + pad
WO_PREFIX = PREFIX_ALIGN
WG_PREFIX = 0     # closure-baked
WU_PREFIX = PREFIX_ALIGN
WD_PREFIX = PREFIX_ALIGN

assert WQ_PREFIX % 64 == 0

# Perchan slot layout per stage:
#   [PREFIX | N_TILE*K i8 weights | N_TILE i32 bias | N_TILE fp32 w_scales]
WQ_SLOT = WQ_PREFIX + N_TILE * D  + N_TILE * 4 + N_TILE * 4    # K=2048
WO_SLOT = WO_PREFIX + N_TILE * QD + N_TILE * 4 + N_TILE * 4    # K=2048 (mh)
WG_SLOT = WG_PREFIX + N_TILE * D  + N_TILE * 4 + N_TILE * 4    # K=2048
WU_SLOT = WU_PREFIX + N_TILE * D  + N_TILE * 4 + N_TILE * 4    # K=2048
WD_SLOT = WD_PREFIX + N_TILE * HD + N_TILE * 4 + N_TILE * 4    # K=8192

N_TILES_Q  = QD // N_TILE               # 512 (was 16 for single-head)
N_TILES_O  = D  // N_TILE               # 512
N_TILES_G  = HD // N_TILE               # 2048
N_TILES_U  = N_TILES_G
N_TILES_D  = D  // N_TILE               # 512

WQ_TOTAL = N_TILES_Q * WQ_SLOT
WO_TOTAL = N_TILES_O * WO_SLOT
WG_TOTAL = N_TILES_G * WG_SLOT
WU_TOTAL = N_TILES_U * WU_SLOT
WD_TOTAL = N_TILES_D * WD_SLOT

GAMMA_BYTES  = D * 2                    # bf16
CS_BYTES     = HEAD_D * 2 * 2           # cos + sin, bf16

# Weights blob layout.
OFF_GAMMA_IN   = 0
OFF_WQ         = OFF_GAMMA_IN   + GAMMA_BYTES
OFF_CS         = OFF_WQ         + WQ_TOTAL
OFF_WO         = OFF_CS         + CS_BYTES
OFF_AF_SCALES  = OFF_WO         + WO_TOTAL
OFF_GAMMA_POST = OFF_AF_SCALES  + AF_SCALES_BYTES
OFF_WG         = OFF_GAMMA_POST + GAMMA_BYTES
OFF_WU         = OFF_WG         + WG_TOTAL
OFF_WD         = OFF_WU         + WU_TOTAL
WEIGHTS_BYTES  = OFF_WD         + WD_TOTAL

# KV cache blob layout: 8 KV heads sequentially. Each KV head has k+v.
PER_KV_HEAD_BYTES = KCACHE_PADDED + VCACHE_PADDED   # 16392
def kv_off_k(h): return h * PER_KV_HEAD_BYTES + 0
def kv_off_v(h): return h * PER_KV_HEAD_BYTES + KCACHE_PADDED
KV_BYTES = N_HEADS_KV * PER_KV_HEAD_BYTES

# Baked scales: only rmsnorm + gate's pair fixed.
def _envf(name, default):
    return float(_os.environ.get(name, default))

ACT_SCALE          = _envf("LAYER_ACT_SCALE",     "0.05")
INV_ACT_SCALE      = _envf("LAYER_INV_ACT_SCALE", str(1.0 / 0.05))
SILU_GATE_SCALE    = _envf("SILU_GATE_SCALE",     "0.05")
GATE_INV_OUT_SCALE = 1.0 / SILU_GATE_SCALE


def _i8(n):
    return np.ndarray[(int(n),), np.dtype[np.int8]]


def _bf16(n):
    from ml_dtypes import bfloat16
    return np.ndarray[(int(n),), np.dtype[bfloat16]]


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
    rt_xin_ty = _i8(D)
    rt_w_ty   = _i8(WEIGHTS_BYTES)
    rt_kv_ty  = _i8(KV_BYTES)
    rt_out_ty = _i8(D)

    t_D_i8        = _i8(D)
    t_QF_i8       = _i8(QF_BYTES)
    t_QR_i8       = _i8(QR_BYTES)
    t_QCHUNKS_HALF_i8 = _i8(QCHUNKS_HALF_BYTES)
    t_QCHUNK_i8   = _i8(QCHUNK_BYTES)
    t_SVCHUNK_i8  = _i8(SVCHUNK_BYTES)
    t_SVCONCAT_HALF_i8 = _i8(SV_CONCAT_HALF)
    t_AFSCALES_i8 = _i8(AF_SCALES_BYTES)
    t_AF_i8       = _i8(AF_BYTES)
    t_HD_i8       = _i8(HD)
    t_UF_i8       = _i8(HD + 8)
    t_D_bf16      = _bf16(D)
    t_CS_bf16     = _bf16(HEAD_D * 2)
    t_WQ_slot     = _i8(WQ_SLOT)
    t_WO_slot     = _i8(WO_SLOT)
    t_WG_slot     = _i8(WG_SLOT)
    t_WU_slot     = _i8(WU_SLOT)
    t_WD_slot     = _i8(WD_SLOT)
    t_KCACHE_i8   = _i8(KCACHE_PADDED)
    t_VCACHE_i8   = _i8(VCACHE_PADDED)

    # --- ObjectFifos ---
    of_xin      = ObjectFifo(t_D_i8,       depth=1, name="xin")

    # Attention side
    of_gam_in   = ObjectFifo(t_D_bf16,     depth=1, name="gam_in")
    of_h1       = ObjectFifo(t_D_i8,       depth=1, name="h1")
    of_wq       = ObjectFifo(t_WQ_slot,    depth=1, name="wq")
    of_qf       = ObjectFifo(t_QF_i8,      depth=1, name="qf")
    of_cs       = ObjectFifo(t_CS_bf16,    depth=1, name="cs")
    of_qr       = ObjectFifo(t_QR_i8,      depth=1, name="qr")
    # q_split writes 2 halves (chunks 0..3 and 4..7) into 2 separate
    # staging fifos -- each memtile then fans out 4 ways, within DMA
    # channel budget.
    of_qchunks_lo = ObjectFifo(t_QCHUNKS_HALF_i8, depth=1, name="qchunks_lo")
    of_qchunks_hi = ObjectFifo(t_QCHUNKS_HALF_i8, depth=1, name="qchunks_hi")
    qchunk_lo_eps = of_qchunks_lo.cons().split(
        offsets=[i * QCHUNK_BYTES for i in range(N_HEADS_KV // 2)],
        obj_types=[t_QCHUNK_i8] * (N_HEADS_KV // 2),
        names=[f"qchunk_{i}" for i in range(N_HEADS_KV // 2)],
    )
    qchunk_hi_eps = of_qchunks_hi.cons().split(
        offsets=[i * QCHUNK_BYTES for i in range(N_HEADS_KV // 2)],
        obj_types=[t_QCHUNK_i8] * (N_HEADS_KV // 2),
        names=[f"qchunk_{N_HEADS_KV // 2 + i}" for i in range(N_HEADS_KV // 2)],
    )
    qchunk_eps = list(qchunk_lo_eps) + list(qchunk_hi_eps)

    # Combined KV per head: [k_header | kcache | v_header | vcache] = 16392 B.
    # To drop shim NoC usage, pack 4 KV heads per shim fill and split via
    # memtile to 4 attn workers. 2 shim fills (lo/hi) instead of 8.
    t_KV_i8       = _i8(KCACHE_PADDED + VCACHE_PADDED)
    KV_HALF_BYTES = (N_HEADS_KV // 2) * (KCACHE_PADDED + VCACHE_PADDED)
    t_KV_HALF_i8  = _i8(KV_HALF_BYTES)
    of_kv_lo = ObjectFifo(t_KV_HALF_i8, depth=1, name="kv_lo")
    of_kv_hi = ObjectFifo(t_KV_HALF_i8, depth=1, name="kv_hi")
    kv_lo_eps = of_kv_lo.cons().split(
        offsets=[i * (KCACHE_PADDED + VCACHE_PADDED)
                 for i in range(N_HEADS_KV // 2)],
        obj_types=[t_KV_i8] * (N_HEADS_KV // 2),
        names=[f"kv_{i}" for i in range(N_HEADS_KV // 2)],
    )
    kv_hi_eps = of_kv_hi.cons().split(
        offsets=[i * (KCACHE_PADDED + VCACHE_PADDED)
                 for i in range(N_HEADS_KV // 2)],
        obj_types=[t_KV_i8] * (N_HEADS_KV // 2),
        names=[f"kv_{N_HEADS_KV // 2 + i}" for i in range(N_HEADS_KV // 2)],
    )
    of_kvs = list(kv_lo_eps) + list(kv_hi_eps)

    # sv-concat split in 2 halves to keep memtile DMA in-channels under 4.
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

    of_svfull   = ObjectFifo(_i8(AF_BYTES), depth=1, name="sv_full")
    of_afscales = ObjectFifo(t_AFSCALES_i8, depth=1, name="af_scales")
    of_af       = ObjectFifo(t_AF_i8,       depth=1, name="af")
    of_wo       = ObjectFifo(t_WO_slot,     depth=1, name="wo")
    of_op       = ObjectFifo(t_D_i8,        depth=1, name="op")

    of_x1       = ObjectFifo(t_D_i8,       depth=1, name="x1")

    # FFN side (unchanged)
    of_gam_post = ObjectFifo(t_D_bf16,     depth=1, name="gam_post")
    of_h2       = ObjectFifo(t_D_i8,       depth=1, name="h2")
    of_wg       = ObjectFifo(t_WG_slot,    depth=1, name="wg")
    of_wu       = ObjectFifo(t_WU_slot,    depth=1, name="wu")
    of_wd       = ObjectFifo(t_WD_slot,    depth=1, name="wd")
    of_gf       = ObjectFifo(t_HD_i8,      depth=1, name="gf")
    of_uf       = ObjectFifo(t_UF_i8,      depth=1, name="uf")
    of_sf       = ObjectFifo(t_HD_i8,      depth=1, name="sf")
    of_df       = ObjectFifo(t_D_i8,       depth=1, name="df")

    of_out      = ObjectFifo(t_D_i8,       depth=1, name="layer_out")

    KO_RMS   = "llama_rmsnorm_int8.cc.o"
    KO_GEMM  = "llama_gemm_int8_srs_tiled_layer.cc.o"
    KO_GEMM2 = "llama_gemm_int8_srs_tiled_layer_mh.cc.o"
    KO_ROPE  = "llama_rope_int8_mh.cc.o"
    KO_SILU  = "llama_silu_mul_int8.cc.o"
    KO_FKV   = "llama_flowkv_mh.cc.o"
    KO_GLUE  = "llama_gqa_glue.cc.o"
    KO_PT    = "llama_layer_pt.cc.o"

    k_rms   = Kernel("llama_rmsnorm_int8", KO_RMS,
                     [t_D_i8, t_D_bf16, t_D_i8, np.float32, np.float32])
    # q_proj-mh
    k_q     = Kernel("llama_gemm_tiled_layer_K2048_N4_perchan_v2_up_q_mh", KO_GEMM2,
                     [t_D_i8, t_WQ_slot, t_QF_i8, np.int32])
    # o_proj-mh
    k_o     = Kernel("llama_gemm_tiled_layer_K2048_N4_perchan_v2_o_mh", KO_GEMM2,
                     [t_AF_i8, t_WO_slot, t_D_i8, np.int32])
    # gate
    k_gate  = Kernel("llama_gemm_tiled_layer_K2048_N4_perchan_gate", KO_GEMM,
                     [t_D_i8, t_WG_slot, t_HD_i8, np.int32,
                      np.float32, np.float32])
    k_up    = Kernel("llama_gemm_tiled_layer_K2048_N4_perchan_v2_up_u", KO_GEMM,
                     [t_D_i8, t_WU_slot, t_UF_i8, np.int32])
    k_down  = Kernel("llama_gemm_tiled_layer_K8192_N4_perchan_v2_d", KO_GEMM,
                     [t_HD_i8, t_WD_slot, t_D_i8, np.int32])
    k_rope  = Kernel("llama_rope_int8_mh_dyn", KO_ROPE,
                     [t_QF_i8, t_CS_bf16, t_QR_i8])
    k_qsplit = Kernel("llama_q_split", KO_GLUE,
                      [t_QR_i8, t_QCHUNKS_HALF_i8, t_QCHUNKS_HALF_i8])
    k_svmerge  = Kernel("llama_sv_merge", KO_GLUE,
                        [t_SVCONCAT_HALF_i8, t_SVCONCAT_HALF_i8, _i8(AF_BYTES)])
    k_afconcat = Kernel("llama_af_concat", KO_GLUE,
                        [_i8(AF_BYTES), t_AFSCALES_i8, t_AF_i8])
    k_fkv   = Kernel("llama_flowkv_mh_kvc", KO_FKV,
                     [t_QCHUNK_i8, t_KV_i8, t_SVCHUNK_i8])
    k_silu  = Kernel("llama_silu_mul_int8_dyn", KO_SILU,
                     [t_HD_i8, t_UF_i8, t_HD_i8])
    k_add   = Kernel("llama_pt_add_D", KO_PT, [t_D_i8, t_D_i8, t_D_i8])

    # --- Worker bodies ---
    def w_rms(c_in, c_gamma, c_out, k):
        x = c_in.acquire(1); g = c_gamma.acquire(1); o = c_out.acquire(1)
        k(x, g, o, ACT_SCALE, INV_ACT_SCALE)
        c_in.release(1); c_gamma.release(1); c_out.release(1)

    def w_q(c_act, c_w, c_out, k):
        a = c_act.acquire(1); o = c_out.acquire(1)
        for t in range_(N_TILES_Q):
            w = c_w.acquire(1); k(a, w, o, _i32(t)); c_w.release(1)
        c_act.release(1); c_out.release(1)

    def w_o(c_act, c_w, c_out, k):
        a = c_act.acquire(1); o = c_out.acquire(1)
        for t in range_(N_TILES_O):
            w = c_w.acquire(1); k(a, w, o, _i32(t)); c_w.release(1)
        c_act.release(1); c_out.release(1)

    def w_gate(c_act, c_w, c_out, k):
        a = c_act.acquire(1); o = c_out.acquire(1)
        for t in range_(N_TILES_G):
            w = c_w.acquire(1)
            k(a, w, o, _i32(t), ACT_SCALE, GATE_INV_OUT_SCALE)
            c_w.release(1)
        c_act.release(1); c_out.release(1)

    def w_up(c_act, c_w, c_out, k):
        a = c_act.acquire(1); o = c_out.acquire(1)
        for t in range_(N_TILES_U):
            w = c_w.acquire(1); k(a, w, o, _i32(t)); c_w.release(1)
        c_act.release(1); c_out.release(1)

    def w_down(c_act, c_w, c_out, k):
        a = c_act.acquire(1); o = c_out.acquire(1)
        for t in range_(N_TILES_D):
            w = c_w.acquire(1); k(a, w, o, _i32(t)); c_w.release(1)
        c_act.release(1); c_out.release(1)

    def w_rope(c_x, c_cs, c_out, k):
        x = c_x.acquire(1); cs = c_cs.acquire(1); o = c_out.acquire(1)
        k(x, cs, o)
        c_x.release(1); c_cs.release(1); c_out.release(1)

    def w_qsplit(c_in, c_lo, c_hi, k):
        x = c_in.acquire(1); lo = c_lo.acquire(1); hi = c_hi.acquire(1)
        k(x, lo, hi)
        c_in.release(1); c_lo.release(1); c_hi.release(1)

    def w_svmerge(c_lo, c_hi, c_out, k):
        lo = c_lo.acquire(1); hi = c_hi.acquire(1); o = c_out.acquire(1)
        k(lo, hi, o)
        c_lo.release(1); c_hi.release(1); c_out.release(1)

    def w_afconcat(c_in, c_sc, c_out, k):
        x = c_in.acquire(1); s = c_sc.acquire(1); o = c_out.acquire(1)
        k(x, s, o)
        c_in.release(1); c_sc.release(1); c_out.release(1)

    def w_attn(c_q, c_kv, c_out, k):
        q = c_q.acquire(1); kv = c_kv.acquire(1); o = c_out.acquire(1)
        k(q, kv, o)
        c_q.release(1); c_kv.release(1); c_out.release(1)

    def w_silu(c_g, c_u, c_out, k):
        g = c_g.acquire(1); u = c_u.acquire(1); o = c_out.acquire(1)
        k(g, u, o)
        c_g.release(1); c_u.release(1); c_out.release(1)

    def w_add(c_a, c_b, c_out, k):
        a = c_a.acquire(1); b = c_b.acquire(1); o = c_out.acquire(1)
        k(a, b, o)
        c_a.release(1); c_b.release(1); c_out.release(1)

    PSK  = 8192   # match layer_d2048 exactly
    ATSK = 16384  # attn workers (flowkv_mh has local arrays + many loops)
    FFNSK = PSK

    workers = [
        # ---- Attention front ----
        Worker(w_rms,    [of_xin.cons(), of_gam_in.cons(), of_h1.prod(), k_rms],
               tile=Tile(0, 2)),
        Worker(w_q,      [of_h1.cons(), of_wq.cons(), of_qf.prod(), k_q],
               tile=Tile(0, 3), stack_size=PSK),
        Worker(w_rope,   [of_qf.cons(), of_cs.cons(), of_qr.prod(), k_rope],
               tile=Tile(0, 4)),
        Worker(w_qsplit, [of_qr.cons(), of_qchunks_lo.prod(),
                          of_qchunks_hi.prod(), k_qsplit],
               tile=Tile(0, 5)),
        # ---- 8 attn workers spread across cols 1-2 ----
        Worker(w_attn, [qchunk_eps[0].cons(), of_kvs[0].cons(),
                        svchunk_eps[0].prod(), k_fkv],
               tile=Tile(1, 2), stack_size=ATSK),
        Worker(w_attn, [qchunk_eps[1].cons(), of_kvs[1].cons(),
                        svchunk_eps[1].prod(), k_fkv],
               tile=Tile(1, 3), stack_size=ATSK),
        Worker(w_attn, [qchunk_eps[2].cons(), of_kvs[2].cons(),
                        svchunk_eps[2].prod(), k_fkv],
               tile=Tile(1, 4), stack_size=ATSK),
        Worker(w_attn, [qchunk_eps[3].cons(), of_kvs[3].cons(),
                        svchunk_eps[3].prod(), k_fkv],
               tile=Tile(1, 5), stack_size=ATSK),
        Worker(w_attn, [qchunk_eps[4].cons(), of_kvs[4].cons(),
                        svchunk_eps[4].prod(), k_fkv],
               tile=Tile(2, 2), stack_size=ATSK),
        Worker(w_attn, [qchunk_eps[5].cons(), of_kvs[5].cons(),
                        svchunk_eps[5].prod(), k_fkv],
               tile=Tile(2, 3), stack_size=ATSK),
        Worker(w_attn, [qchunk_eps[6].cons(), of_kvs[6].cons(),
                        svchunk_eps[6].prod(), k_fkv],
               tile=Tile(2, 4), stack_size=ATSK),
        Worker(w_attn, [qchunk_eps[7].cons(), of_kvs[7].cons(),
                        svchunk_eps[7].prod(), k_fkv],
               tile=Tile(2, 5), stack_size=ATSK),
        # ---- Attention back ----
        Worker(w_svmerge,  [of_svconcat_lo.cons(), of_svconcat_hi.cons(),
                            of_svfull.prod(), k_svmerge],
               tile=Tile(3, 2)),
        Worker(w_afconcat, [of_svfull.cons(), of_afscales.cons(),
                            of_af.prod(), k_afconcat],
               tile=Tile(3, 5)),
        Worker(w_o,    [of_af.cons(), of_wo.cons(), of_op.prod(), k_o],
               tile=Tile(3, 3), stack_size=PSK),
        Worker(w_add,  [of_op.cons(), of_xin.cons(), of_x1.prod(), k_add],
               tile=Tile(3, 4)),
        # ---- FFN (match aie2_layer_d2048 placement exactly) ----
        Worker(w_rms,  [of_x1.cons(), of_gam_post.cons(), of_h2.prod(), k_rms],
               tile=Tile(6, 4)),
        Worker(w_gate, [of_h2.cons(), of_wg.cons(), of_gf.prod(), k_gate],
               tile=Tile(4, 2), stack_size=FFNSK),
        Worker(w_up,   [of_h2.cons(), of_wu.cons(), of_uf.prod(), k_up],
               tile=Tile(5, 2), stack_size=FFNSK),
        Worker(w_silu, [of_gf.cons(), of_uf.cons(), of_sf.prod(), k_silu],
               tile=Tile(4, 5), stack_size=FFNSK),
        Worker(w_down, [of_sf.cons(), of_wd.cons(), of_df.prod(), k_down],
               tile=Tile(6, 2), stack_size=FFNSK),
        Worker(w_add,  [of_df.cons(), of_x1.cons(), of_out.prod(), k_add],
               tile=Tile(7, 5)),
    ]

    rt = Runtime()
    with rt.sequence(rt_xin_ty, rt_w_ty, rt_kv_ty, rt_out_ty) as (
        xin, wblob, kvblob, out
    ):
        rt.start(*workers)

        tgs = []
        def add_fill(prod, src, off, slot_bytes, n_slots, total):
            tg = rt.task_group(); tgs.append(tg)
            rt.fill(prod, src,
                    tap=strided_tap(total, off, slot_bytes, slot_bytes, n_slots),
                    task_group=tg)

        xin_tg = rt.task_group(); tgs.append(xin_tg)
        rt.fill(of_xin.prod(), xin, task_group=xin_tg)

        add_fill(of_gam_in.prod(),   wblob, OFF_GAMMA_IN,   GAMMA_BYTES,    1,         WEIGHTS_BYTES)
        add_fill(of_wq.prod(),       wblob, OFF_WQ,         WQ_SLOT,        N_TILES_Q, WEIGHTS_BYTES)
        add_fill(of_cs.prod(),       wblob, OFF_CS,         CS_BYTES,       1,         WEIGHTS_BYTES)
        add_fill(of_wo.prod(),       wblob, OFF_WO,         WO_SLOT,        N_TILES_O, WEIGHTS_BYTES)
        add_fill(of_afscales.prod(), wblob, OFF_AF_SCALES,  AF_SCALES_BYTES,1,         WEIGHTS_BYTES)
        add_fill(of_gam_post.prod(), wblob, OFF_GAMMA_POST, GAMMA_BYTES,    1,         WEIGHTS_BYTES)
        add_fill(of_wg.prod(),       wblob, OFF_WG,         WG_SLOT,        N_TILES_G, WEIGHTS_BYTES)
        add_fill(of_wu.prod(),       wblob, OFF_WU,         WU_SLOT,        N_TILES_U, WEIGHTS_BYTES)
        add_fill(of_wd.prod(),       wblob, OFF_WD,         WD_SLOT,        N_TILES_D, WEIGHTS_BYTES)

        # 2 KV fills (4 heads each).
        add_fill(of_kv_lo.prod(), kvblob, 0,                          KV_HALF_BYTES, 1, KV_BYTES)
        add_fill(of_kv_hi.prod(), kvblob, KV_HALF_BYTES,              KV_HALF_BYTES, 1, KV_BYTES)

        out_tg = rt.task_group(); tgs.append(out_tg)
        rt.drain(of_out.cons(), out, wait=True, task_group=out_tg)

        for tg in tgs:
            rt.finish_task_group(tg)

    return Program(NPU2(), rt).resolve_program()


def main():
    argparse.ArgumentParser().parse_args(sys.argv[1:])
    print(build())


if __name__ == "__main__":
    main()
