"""Phase 6c.5b.5: N-layer decode chain at production Llama 3.2 1B shapes
(D=2048, HD=8192, single-head, T=128) with per-channel weight quant +
per-token dynamic activation scales. Combines 6c.3 (chain pattern from
aie2_chain_real) with 6c.5b.4 (perchan + dynamic-scale single-layer).

Topology: 14 workers = 6c.5b.4's 13 single-layer workers + 1 router.
Each worker loops N_LAYERS times. Per-layer weights stream via per-fifo
BD-chained TAPs.

Residual loop-back (mirrors chain_real):
  - L=0:      seed (from runtime xin) -> router -> routed
  - L=1..N-1: back (from prev layer's add2) -> router -> routed
  - L=N-1:    add2 writes to `out` (runtime drain) instead of `back`
  routed broadcasts to rmsnorm1 AND add1 (residual hold).

Per-fifo wblob layout: each fifo's per-layer slots are contiguous, e.g.
all WQ slots for L=0..N-1 packed back to back. TAP walks that section as
[1, N_LAYERS * N_TILES_K, outer, inner]. Same recipe as 6c.5b.4's
single-layer slot layout (64 B prefix + perchan body), repeated N_LAYERS
times per fifo.

All ObjectFifos at depth=1 (Bug 8 audit -- same constraint as 6c.5b.4).
All slot prefixes 64 B (Bug 7 alignment).

Known limit: N_LAYERS <= 3 at production shapes. At N >= 4 the single
BD-chained TAP per fifo exceeds the shim-NoC BD's effective wrap-size
limit (HW seems to truncate dim sizes silently while the verifier
exempts contiguous transfers from the 10-bit check). Per-layer rt.fill
loops sidestep the wrap limit but then hit the 4-deep per-channel task
queue (Bug 3b) and even with inline finish_task_group between batches
the chain still corrupts. Real fix is memtile-staged ObjectFifo (shim
does a single linear DMA into memtile; memtile re-DMAs strided into
CT) -- to file as 6c.5b.5b. For now N<=3 validates the per-layer
dynamic-scale plumbing.
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


# --- Production shapes (mirror aie2_layer_d2048) ---
D       = 2048
HD      = 8192
HEAD_D  = 64
N_HEADS = 1
N_KV    = 1
QD      = N_HEADS * HEAD_D       # 64
KVD     = N_KV    * HEAD_D       # 64
T       = int(_os.environ.get("LLAMA_CHAIN_T", "128"))
N_TILE  = 4
N_LAYERS = int(_os.environ.get("LLAMA_CHAIN_N", "2"))

# Slot prefixes (mirror layer_d2048).
PREFIX_ALIGN = 64
WQ_PREFIX = PREFIX_ALIGN   # v2_up_q
WO_PREFIX = PREFIX_ALIGN   # v2_o
WG_PREFIX = 0              # closure _perchan (LUT-locked)
WU_PREFIX = PREFIX_ALIGN   # v2_up_u
WD_PREFIX = PREFIX_ALIGN   # v2_d

WQ_SLOT = WQ_PREFIX + N_TILE * D  + N_TILE * 4 + N_TILE * 4
WO_SLOT = WO_PREFIX + N_TILE * QD + N_TILE * 4 + N_TILE * 4
WG_SLOT = WG_PREFIX + N_TILE * D  + N_TILE * 4 + N_TILE * 4
WU_SLOT = WU_PREFIX + N_TILE * D  + N_TILE * 4 + N_TILE * 4
WD_SLOT = WD_PREFIX + N_TILE * HD + N_TILE * 4 + N_TILE * 4

N_TILES_Q = QD // N_TILE          # 16
N_TILES_O = D  // N_TILE          # 512
N_TILES_G = HD // N_TILE          # 2048
N_TILES_U = N_TILES_G
N_TILES_D = D  // N_TILE          # 512

# Per-fifo total slot count across all layers (used by TAP n_slots).
TOT_GAMMA = N_LAYERS
TOT_CS    = N_LAYERS
TOT_KV    = N_LAYERS
TOT_WQ    = N_LAYERS * N_TILES_Q
TOT_WO    = N_LAYERS * N_TILES_O
TOT_WG    = N_LAYERS * N_TILES_G
TOT_WU    = N_LAYERS * N_TILES_U
TOT_WD    = N_LAYERS * N_TILES_D

GAMMA_BYTES   = D * 2
CS_BYTES      = HEAD_D * 2 * 2

# Fifo tail sizes (downstream piggyback).
QF_BYTES = QD + 8
QR_BYTES = QD + 8
UF_BYTES = HD + 8

# KV cache with 8 B header per layer.
KV_HEADER = 8
KCACHE_BYTES = T * KVD
VCACHE_BYTES = T * KVD
KCACHE_PADDED = KV_HEADER + KCACHE_BYTES
VCACHE_PADDED = KV_HEADER + VCACHE_BYTES

# Per-fifo total bytes (contiguous in wblob).
GAMMA_TOTAL = TOT_GAMMA * GAMMA_BYTES
CS_TOTAL    = TOT_CS    * CS_BYTES
WQ_TOTAL    = TOT_WQ    * WQ_SLOT
WO_TOTAL    = TOT_WO    * WO_SLOT
WG_TOTAL    = TOT_WG    * WG_SLOT
WU_TOTAL    = TOT_WU    * WU_SLOT
WD_TOTAL    = TOT_WD    * WD_SLOT

# wblob layout (per-fifo sections contiguous across layers).
OFF_GAMMA_IN   = 0
OFF_WQ         = OFF_GAMMA_IN   + GAMMA_TOTAL
OFF_CS         = OFF_WQ         + WQ_TOTAL
OFF_WO         = OFF_CS         + CS_TOTAL
OFF_GAMMA_POST = OFF_WO         + WO_TOTAL
OFF_WG         = OFF_GAMMA_POST + GAMMA_TOTAL
OFF_WU         = OFF_WG         + WG_TOTAL
OFF_WD         = OFF_WU         + WU_TOTAL
WEIGHTS_BYTES  = OFF_WD         + WD_TOTAL

# kvblob layout (per-layer KCACHE_PADDED followed by per-layer VCACHE_PADDED,
# interleaved per layer so a single TAP per cache walks N_LAYERS slots
# with per-layer stride = KCACHE_PADDED + VCACHE_PADDED).
PER_LAYER_KV = KCACHE_PADDED + VCACHE_PADDED
OFF_K        = 0
OFF_V        = OFF_K + KCACHE_PADDED
KV_BYTES     = N_LAYERS * PER_LAYER_KV

# Baked scales (same as layer_d2048).
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

    t_D_i8       = _i8(D)
    t_QF_i8      = _i8(QF_BYTES)
    t_QR_i8      = _i8(QR_BYTES)
    t_AF_i8      = _i8(QD)
    t_HD_i8      = _i8(HD)
    t_UF_i8      = _i8(UF_BYTES)
    t_D_bf16     = _bf16(D)
    t_CS_bf16    = _bf16(HEAD_D * 2)
    t_WQ_slot    = _i8(WQ_SLOT)
    t_WO_slot    = _i8(WO_SLOT)
    t_WG_slot    = _i8(WG_SLOT)
    t_WU_slot    = _i8(WU_SLOT)
    t_WD_slot    = _i8(WD_SLOT)
    t_KCACHE_i8  = _i8(KCACHE_PADDED)
    t_VCACHE_i8  = _i8(VCACHE_PADDED)
    t_PROBS_fp32 = np.ndarray[(T,), np.dtype[np.float32]]

    # --- ObjectFifos (all depth=1; Bug 8) ---
    # Residual loop-back.
    of_seed   = ObjectFifo(t_D_i8, depth=1, name="seed")
    of_back   = ObjectFifo(t_D_i8, depth=1, name="back")
    of_routed = ObjectFifo(t_D_i8, depth=1, name="routed")  # broadcast to rmsnorm1 + add1
    of_out    = ObjectFifo(t_D_i8, depth=1, name="layer_out")

    # Attention side
    of_gam_in   = ObjectFifo(t_D_bf16,     depth=1, name="gam_in")
    of_h1       = ObjectFifo(t_D_i8,       depth=1, name="h1")
    of_wq       = ObjectFifo(t_WQ_slot,    depth=1, name="wq")
    of_qf       = ObjectFifo(t_QF_i8,      depth=1, name="qf")
    of_cs       = ObjectFifo(t_CS_bf16,    depth=1, name="cs")
    of_qr       = ObjectFifo(t_QR_i8,      depth=1, name="qr")
    of_kcache   = ObjectFifo(t_KCACHE_i8,  depth=1, name="kcache")
    of_vcache   = ObjectFifo(t_VCACHE_i8,  depth=1, name="vcache")
    of_probs    = ObjectFifo(t_PROBS_fp32, depth=1, name="probs")
    of_af       = ObjectFifo(t_AF_i8,      depth=1, name="af")
    of_wo       = ObjectFifo(t_WO_slot,    depth=1, name="wo")
    of_op       = ObjectFifo(t_D_i8,       depth=1, name="op")

    # Residual link 1 (broadcast: rmsnorm2 + add2)
    of_x1       = ObjectFifo(t_D_i8,       depth=1, name="x1")

    # FFN side
    of_gam_post = ObjectFifo(t_D_bf16,     depth=1, name="gam_post")
    of_h2       = ObjectFifo(t_D_i8,       depth=1, name="h2")  # broadcast: gate + up
    of_wg       = ObjectFifo(t_WG_slot,    depth=1, name="wg")
    of_wu       = ObjectFifo(t_WU_slot,    depth=1, name="wu")
    of_wd       = ObjectFifo(t_WD_slot,    depth=1, name="wd")
    of_gf       = ObjectFifo(t_HD_i8,      depth=1, name="gf")
    of_uf       = ObjectFifo(t_UF_i8,      depth=1, name="uf")
    of_sf       = ObjectFifo(t_HD_i8,      depth=1, name="sf")
    of_df       = ObjectFifo(t_D_i8,       depth=1, name="df")

    KO_RMS  = "llama_rmsnorm_int8.cc.o"
    KO_GEMM = "llama_gemm_int8_srs_tiled_layer.cc.o"
    KO_ROPE = "llama_rope_int8.cc.o"
    KO_SILU = "llama_silu_mul_int8.cc.o"
    KO_FKV  = "llama_flowkv.cc.o"
    KO_PT   = "llama_layer_pt.cc.o"

    k_rms   = Kernel("llama_rmsnorm_int8", KO_RMS,
                     [t_D_i8, t_D_bf16, t_D_i8, np.float32, np.float32])
    k_q     = Kernel("llama_gemm_tiled_layer_K2048_N4_perchan_v2_up_q", KO_GEMM,
                     [t_D_i8, t_WQ_slot, t_QF_i8, np.int32])
    k_o     = Kernel("llama_gemm_tiled_layer_K64_N4_perchan_v2_o", KO_GEMM,
                     [t_AF_i8, t_WO_slot, t_D_i8, np.int32])
    k_gate  = Kernel("llama_gemm_tiled_layer_K2048_N4_perchan_gate", KO_GEMM,
                     [t_D_i8, t_WG_slot, t_HD_i8, np.int32,
                      np.float32, np.float32])
    k_up    = Kernel("llama_gemm_tiled_layer_K2048_N4_perchan_v2_up_u", KO_GEMM,
                     [t_D_i8, t_WU_slot, t_UF_i8, np.int32])
    k_down  = Kernel("llama_gemm_tiled_layer_K8192_N4_perchan_v2_d", KO_GEMM,
                     [t_HD_i8, t_WD_slot, t_D_i8, np.int32])
    k_rope  = Kernel("llama_rope_int8_dyn", KO_ROPE,
                     [t_QF_i8, t_CS_bf16, t_QR_i8])
    k_qk    = Kernel("llama_flowkv_qk_dyn", KO_FKV,
                     [t_QR_i8, t_KCACHE_i8, t_PROBS_fp32])
    k_sv    = Kernel("llama_flowkv_sv_dyn", KO_FKV,
                     [t_VCACHE_i8, t_PROBS_fp32, t_AF_i8])
    k_silu  = Kernel("llama_silu_mul_int8_dyn", KO_SILU,
                     [t_HD_i8, t_UF_i8, t_HD_i8])
    k_add   = Kernel("llama_pt_add_D", KO_PT, [t_D_i8, t_D_i8, t_D_i8])

    # --- Worker bodies (loop N_LAYERS times) ---

    # Router: per-layer 2-in (seed OR back) -> 1-out (routed). First iter
    # reads seed; subsequent iters read back. Peeled to avoid a Python
    # conditional on a range_ loop var.
    # MUST use range_(D) (MLIR scf.for) instead of Python range(D) for
    # the inner copy: at D=2048 Python unrolls into 2048 inline stores
    # which overflows the AIE2P core's program memory.
    def w_router(c_seed, c_back, p_routed):
        x = c_seed.acquire(1); o = p_routed.acquire(1)
        for i in range_(D):
            o[i] = x[i]
        p_routed.release(1); c_seed.release(1)
        for _ in range_(N_LAYERS - 1):
            x = c_back.acquire(1); o = p_routed.acquire(1)
            for i in range_(D):
                o[i] = x[i]
            p_routed.release(1); c_back.release(1)

    def w_rms(c_in, c_gamma, c_out, k):
        for _ in range_(N_LAYERS):
            x = c_in.acquire(1); g = c_gamma.acquire(1); o = c_out.acquire(1)
            k(x, g, o, ACT_SCALE, INV_ACT_SCALE)
            c_in.release(1); c_gamma.release(1); c_out.release(1)

    def w_q(c_act, c_w, c_out, k):
        for _ in range_(N_LAYERS):
            a = c_act.acquire(1); o = c_out.acquire(1)
            for t in range_(N_TILES_Q):
                w = c_w.acquire(1); k(a, w, o, _i32(t)); c_w.release(1)
            c_act.release(1); c_out.release(1)

    def w_o(c_act, c_w, c_out, k):
        for _ in range_(N_LAYERS):
            a = c_act.acquire(1); o = c_out.acquire(1)
            for t in range_(N_TILES_O):
                w = c_w.acquire(1); k(a, w, o, _i32(t)); c_w.release(1)
            c_act.release(1); c_out.release(1)

    def w_gate(c_act, c_w, c_out, k):
        for _ in range_(N_LAYERS):
            a = c_act.acquire(1); o = c_out.acquire(1)
            for t in range_(N_TILES_G):
                w = c_w.acquire(1)
                k(a, w, o, _i32(t), ACT_SCALE, GATE_INV_OUT_SCALE)
                c_w.release(1)
            c_act.release(1); c_out.release(1)

    def w_up(c_act, c_w, c_out, k):
        for _ in range_(N_LAYERS):
            a = c_act.acquire(1); o = c_out.acquire(1)
            for t in range_(N_TILES_U):
                w = c_w.acquire(1); k(a, w, o, _i32(t)); c_w.release(1)
            c_act.release(1); c_out.release(1)

    def w_down(c_act, c_w, c_out, k):
        for _ in range_(N_LAYERS):
            a = c_act.acquire(1); o = c_out.acquire(1)
            for t in range_(N_TILES_D):
                w = c_w.acquire(1); k(a, w, o, _i32(t)); c_w.release(1)
            c_act.release(1); c_out.release(1)

    def w_rope(c_x, c_cs, c_out, k):
        for _ in range_(N_LAYERS):
            x = c_x.acquire(1); cs = c_cs.acquire(1); o = c_out.acquire(1)
            k(x, cs, o)
            c_x.release(1); c_cs.release(1); c_out.release(1)

    def w_qk(c_q, c_k, c_probs, k):
        for _ in range_(N_LAYERS):
            q = c_q.acquire(1); kk = c_k.acquire(1); p = c_probs.acquire(1)
            k(q, kk, p)
            c_q.release(1); c_k.release(1); c_probs.release(1)

    def w_sv(c_v, c_probs, c_out, k):
        for _ in range_(N_LAYERS):
            v = c_v.acquire(1); p = c_probs.acquire(1); o = c_out.acquire(1)
            k(v, p, o)
            c_v.release(1); c_probs.release(1); c_out.release(1)

    def w_silu(c_g, c_u, c_out, k):
        for _ in range_(N_LAYERS):
            g = c_g.acquire(1); u = c_u.acquire(1); o = c_out.acquire(1)
            k(g, u, o)
            c_g.release(1); c_u.release(1); c_out.release(1)

    def w_add1(c_a, c_b, c_out, k):
        for _ in range_(N_LAYERS):
            a = c_a.acquire(1); b = c_b.acquire(1); o = c_out.acquire(1)
            k(a, b, o)
            c_a.release(1); c_b.release(1); c_out.release(1)

    # add2: writes to `back` for L=0..N-2 (residual loop), `out` for L=N-1.
    def w_add2(c_a, c_b, c_back, c_out, k):
        for _ in range_(N_LAYERS - 1):
            a = c_a.acquire(1); b = c_b.acquire(1); o = c_back.acquire(1)
            k(a, b, o)
            c_a.release(1); c_b.release(1); c_back.release(1)
        a = c_a.acquire(1); b = c_b.acquire(1); o = c_out.acquire(1)
        k(a, b, o)
        c_a.release(1); c_b.release(1); c_out.release(1)

    PSK = 8192
    workers = [
        # Router (residual loop-back driver).
        Worker(w_router,
               [of_seed.cons(), of_back.cons(), of_routed.prod()],
               tile=Tile(7, 4)),

        # Attention side
        Worker(w_rms, [of_routed.cons(), of_gam_in.cons(), of_h1.prod(), k_rms],
               tile=Tile(5, 4)),
        Worker(w_q,   [of_h1.cons(), of_wq.cons(), of_qf.prod(), k_q],
               tile=Tile(0, 2), stack_size=PSK),
        Worker(w_rope, [of_qf.cons(), of_cs.cons(), of_qr.prod(), k_rope],
               tile=Tile(4, 4)),
        Worker(w_qk,  [of_qr.cons(), of_kcache.cons(), of_probs.prod(), k_qk],
               tile=Tile(0, 4), stack_size=16384),
        Worker(w_sv,  [of_vcache.cons(), of_probs.cons(), of_af.prod(), k_sv],
               tile=Tile(0, 5)),
        Worker(w_o,   [of_af.cons(), of_wo.cons(), of_op.prod(), k_o],
               tile=Tile(3, 2), stack_size=PSK),
        Worker(w_add1, [of_op.cons(), of_routed.cons(), of_x1.prod(), k_add],
               tile=Tile(6, 5)),

        # FFN side
        Worker(w_rms, [of_x1.cons(), of_gam_post.cons(), of_h2.prod(), k_rms],
               tile=Tile(6, 4)),
        Worker(w_gate, [of_h2.cons(), of_wg.cons(), of_gf.prod(), k_gate],
               tile=Tile(4, 2), stack_size=PSK),
        Worker(w_up,   [of_h2.cons(), of_wu.cons(), of_uf.prod(), k_up],
               tile=Tile(5, 2), stack_size=PSK),
        Worker(w_silu, [of_gf.cons(), of_uf.cons(), of_sf.prod(), k_silu],
               tile=Tile(4, 5), stack_size=PSK),
        Worker(w_down, [of_sf.cons(), of_wd.cons(), of_df.prod(), k_down],
               tile=Tile(6, 2), stack_size=PSK),

        # add2 with peeled final iter to route output to runtime drain.
        Worker(w_add2,
               [of_df.cons(), of_x1.cons(), of_back.prod(), of_out.prod(), k_add],
               tile=Tile(7, 5)),
    ]

    rt = Runtime()
    with rt.sequence(rt_xin_ty, rt_w_ty, rt_kv_ty, rt_out_ty) as (
        xin, wblob, kvblob, out
    ):
        rt.start(*workers)

        tgs = []
        def add_fill_n(prod, src, off, slot_bytes, n_slots, total):
            tg = rt.task_group(); tgs.append(tg)
            rt.fill(prod, src,
                    tap=strided_tap(total, off, slot_bytes, slot_bytes, n_slots),
                    task_group=tg)
        # Compatibility alias for the older code path's call sites.
        def add_fill_per_layer(prod, src, base_off, per_layer_stride,
                               slot_bytes, slots_per_layer, total):
            # Single BD-chained TAP per fifo covering ALL N_LAYERS * slots_per_layer
            # slots. Hits the AIE2P shim BD wrap-size limit at N >= 4 for
            # production-shape slots, but kept this way for N=2/3 (single
            # task per channel; no Bug 3b queue pressure).
            add_fill_n(prod, src, base_off, slot_bytes,
                       N_LAYERS * slots_per_layer, total)

        # Runtime activation in -> seed (1 slot, layer 0's input).
        seed_tg = rt.task_group(); tgs.append(seed_tg)
        rt.fill(of_seed.prod(), xin, task_group=seed_tg)

        # Per-fifo, per-layer rt.fill (N_LAYERS calls per fifo). Each fill's
        # TAP covers slots_per_layer slots for one layer.
        add_fill_per_layer(of_gam_in.prod(),   wblob, OFF_GAMMA_IN,
                           GAMMA_BYTES, GAMMA_BYTES, 1, WEIGHTS_BYTES)
        add_fill_per_layer(of_wq.prod(),       wblob, OFF_WQ,
                           N_TILES_Q * WQ_SLOT, WQ_SLOT, N_TILES_Q, WEIGHTS_BYTES)
        add_fill_per_layer(of_cs.prod(),       wblob, OFF_CS,
                           CS_BYTES, CS_BYTES, 1, WEIGHTS_BYTES)
        add_fill_per_layer(of_wo.prod(),       wblob, OFF_WO,
                           N_TILES_O * WO_SLOT, WO_SLOT, N_TILES_O, WEIGHTS_BYTES)
        add_fill_per_layer(of_gam_post.prod(), wblob, OFF_GAMMA_POST,
                           GAMMA_BYTES, GAMMA_BYTES, 1, WEIGHTS_BYTES)
        add_fill_per_layer(of_wg.prod(),       wblob, OFF_WG,
                           N_TILES_G * WG_SLOT, WG_SLOT, N_TILES_G, WEIGHTS_BYTES)
        add_fill_per_layer(of_wu.prod(),       wblob, OFF_WU,
                           N_TILES_U * WU_SLOT, WU_SLOT, N_TILES_U, WEIGHTS_BYTES)
        add_fill_per_layer(of_wd.prod(),       wblob, OFF_WD,
                           N_TILES_D * WD_SLOT, WD_SLOT, N_TILES_D, WEIGHTS_BYTES)

        # KV: single TAP per cache walking N_LAYERS slots with
        # stride PER_LAYER_KV.
        kc_tg = rt.task_group(); tgs.append(kc_tg)
        rt.fill(of_kcache.prod(), kvblob,
                tap=strided_tap(KV_BYTES, OFF_K, PER_LAYER_KV,
                                KCACHE_PADDED, N_LAYERS),
                task_group=kc_tg)
        vc_tg = rt.task_group(); tgs.append(vc_tg)
        rt.fill(of_vcache.prod(), kvblob,
                tap=strided_tap(KV_BYTES, OFF_V, PER_LAYER_KV,
                                VCACHE_PADDED, N_LAYERS),
                task_group=vc_tg)

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
