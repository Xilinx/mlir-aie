"""Phase 6c.5b.3: Attention-half at production Llama 3.2 1B (D=2048,
single-head, T=128) with per-channel weight quant + per-token dynamic
activation scales delivered via slot prefixes / buffer tails.

Seven-worker xclbin wiring (unchanged from 6c.3b.2):
  rmsnorm1 -> q_proj (tiled, perchan) -> rope_q -> flowkv_qk -> flowkv_sv
                                                                  \\-> o_proj (tiled, perchan) -> add1

Single-head simplification: N_HEADS = N_KV = 1, HEAD_DIM = 64. The full
KV cache is supplied from runtime. T capped at 128 (flowkv stack budget).

Dynamic-scale plumbing (vs 6c.3b.2's closure-baked scales):
- WQ slot prefix (64 B): act_scale + inv_out_scale + q_out_scale + spare
  + pad. q_proj_v2_up kernel mirrors the trailing 8 B into of_qf's tail.
- rope_dyn reads act_scale from qf tail, copies the 8 B through to qr.
- of_kcache / of_vcache buffers carry 8 B headers (k_scale + spare;
  v_scale + inv_sv_out_scale). flowkv_*_dyn variants read from headers.
- WO slot prefix (64 B): act_scale (= sv_out) + inv_out_scale (= 1/ACT_SCALE).
- All fifos at depth=1 (Bug 8 -- same-xclbin multi-dispatch desync).
- Prefix size 64 B is the minimum that preserves aie::load_v<64> weight
  alignment (Bug 7).
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


# --- Production Llama 3.2 1B attention shapes (single-head subset) ---
D       = 2048
HEAD_D  = 64
N_HEADS = 1
N_KV    = 1
QD      = N_HEADS * HEAD_D       # 64
KVD     = N_KV    * HEAD_D       # 64
T       = int(_os.environ.get("LLAMA_ATTN_T", "128"))
N_TILE  = 4

# Slot prefix is 64 B (multiple of 64 keeps weight body aligned for
# aie::load_v<64>, per Bug 7). q_proj uses _perchan_v2_up: bytes
# [0..8] = (act_scale, inv_out_scale); [8..16] = (q_out_scale, spare).
# o_proj uses _perchan_v2: bytes [0..8] = (act_scale, inv_out_scale).
PREFIX_ALIGN = 64
WQ_PREFIX = PREFIX_ALIGN
WO_PREFIX = PREFIX_ALIGN
# Perchan slot layout: [PREFIX | N_TILE*K weights | N_TILE i32 bias |
#                       N_TILE fp32 w_scales]
WQ_SLOT = WQ_PREFIX + N_TILE * D  + N_TILE * 4 + N_TILE * 4
WO_SLOT = WO_PREFIX + N_TILE * QD + N_TILE * 4 + N_TILE * 4

N_TILES_Q = QD // N_TILE          # 16
N_TILES_O = D  // N_TILE          # 512

WQ_TOTAL = N_TILES_Q * WQ_SLOT
WO_TOTAL = N_TILES_O * WO_SLOT

GAMMA_BYTES  = D * 2              # bf16
CS_BYTES     = HEAD_D * 2 * 2     # cos + sin, bf16

# qf / qr carry 8 B tails for the per-token q_scale (and a spare fp32);
# rope_dyn passes them through.
QF_BYTES = QD + 8
QR_BYTES = QD + 8

# kcache / vcache buffers carry 8 B leading headers. k_header =
# (k_scale fp32, spare); v_header = (v_scale fp32, inv_sv_out fp32).
KV_HEADER = 8
KCACHE_BYTES = T * KVD
VCACHE_BYTES = T * KVD
KCACHE_PADDED = KV_HEADER + KCACHE_BYTES
VCACHE_PADDED = KV_HEADER + VCACHE_BYTES

# Weights blob layout.
OFF_GAMMA = 0
OFF_WQ    = OFF_GAMMA + GAMMA_BYTES
OFF_CS    = OFF_WQ    + WQ_TOTAL
OFF_WO    = OFF_CS    + CS_BYTES
WEIGHTS_BYTES = OFF_WO + WO_TOTAL

# KV cache blob layout.
OFF_K    = 0
OFF_V    = OFF_K + KCACHE_PADDED
KV_BYTES = OFF_V + VCACHE_PADDED

# Baked-at-build-time scales. Only rmsnorm's pair is fully fixed; all
# other scales are dynamic (slot prefix / KV header / tail).
ACT_SCALE        = 0.05
INV_ACT_SCALE    = 1.0 / ACT_SCALE


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
    t_QF_i8      = _i8(QF_BYTES)         # QD + 8 (rope tail)
    t_QR_i8      = _i8(QR_BYTES)         # QD + 8 (qk tail)
    t_AF_i8      = _i8(QD)               # sv output (no tail needed; goes to o_proj act)
    t_KCACHE_i8  = _i8(KCACHE_PADDED)    # 8 + T*KVD
    t_VCACHE_i8  = _i8(VCACHE_PADDED)    # 8 + T*KVD
    t_D_bf16     = _bf16(D)
    t_CS_bf16    = _bf16(HEAD_D * 2)
    t_WQ_slot    = _i8(WQ_SLOT)
    t_WO_slot    = _i8(WO_SLOT)
    t_PROBS_fp32 = np.ndarray[(T,), np.dtype[np.float32]]

    # --- ObjectFifos (all depth=1; Bug 8) ---
    of_xin    = ObjectFifo(t_D_i8,        depth=1, name="xin")  # broadcast to rmsnorm + add1
    of_h1     = ObjectFifo(t_D_i8,        depth=1, name="h1")
    of_gamma  = ObjectFifo(t_D_bf16,      depth=1, name="gamma")
    of_wq     = ObjectFifo(t_WQ_slot,     depth=1, name="wq")
    of_qf     = ObjectFifo(t_QF_i8,       depth=1, name="qf")
    of_cs     = ObjectFifo(t_CS_bf16,     depth=1, name="cs")
    of_qr     = ObjectFifo(t_QR_i8,       depth=1, name="qr")
    of_kcache = ObjectFifo(t_KCACHE_i8,   depth=1, name="kcache")
    of_vcache = ObjectFifo(t_VCACHE_i8,   depth=1, name="vcache")
    of_probs  = ObjectFifo(t_PROBS_fp32,  depth=1, name="probs")
    of_af     = ObjectFifo(t_AF_i8,       depth=1, name="af")
    of_wo     = ObjectFifo(t_WO_slot,     depth=1, name="wo")
    of_op     = ObjectFifo(t_D_i8,        depth=1, name="op")
    of_out    = ObjectFifo(t_D_i8,        depth=1, name="layer_out")

    KO_RMS  = "llama_rmsnorm_int8.cc.o"
    KO_GEMM = "llama_gemm_int8_srs_tiled_attn.cc.o"
    KO_ROPE = "llama_rope_int8.cc.o"
    KO_FKV  = "llama_flowkv.cc.o"
    KO_PT   = "llama_layer_pt.cc.o"

    k_rms   = Kernel("llama_rmsnorm_int8", KO_RMS,
                     [t_D_i8, t_D_bf16, t_D_i8, np.float32, np.float32])
    # q_proj: _perchan_v2_up. Slot is WQ_SLOT bytes (64 B prefix + perchan
    # body). Output is QF_BYTES = QD+8 -- kernel mirrors slot[8..16]
    # into out[QD..QD+8] each iter.
    k_q     = Kernel("llama_gemm_tiled_attn_K2048_N4_perchan_v2_up", KO_GEMM,
                     [t_D_i8, t_WQ_slot, t_QF_i8, np.int32])
    # o_proj: _perchan_v2. Slot WO_SLOT (64 B prefix). 4 args.
    k_o     = Kernel("llama_gemm_tiled_attn_K64_N4_perchan_v2", KO_GEMM,
                     [t_AF_i8, t_WO_slot, t_D_i8, np.int32])
    # rope_dyn: reads act_scale from x[QD..QD+4]; passes 8 B tail to out.
    k_rope  = Kernel("llama_rope_int8_dyn", KO_ROPE,
                     [t_QF_i8, t_CS_bf16, t_QR_i8])
    # flowkv_qk_dyn: q sized QD+8 (q_scale tail), k sized KCACHE_PADDED
    # (k_scale header). 3 args.
    k_qk    = Kernel("llama_flowkv_qk_dyn", KO_FKV,
                     [t_QR_i8, t_KCACHE_i8, t_PROBS_fp32])
    # flowkv_sv_dyn: v sized VCACHE_PADDED (v_scale + inv_out_scale header). 3 args.
    k_sv    = Kernel("llama_flowkv_sv_dyn", KO_FKV,
                     [t_VCACHE_i8, t_PROBS_fp32, t_AF_i8])
    k_add   = Kernel("llama_pt_add_D", KO_PT, [t_D_i8, t_D_i8, t_D_i8])

    # --- Worker bodies ---
    def w_rms(c_in, c_gamma, c_out, k):
        x = c_in.acquire(1); g = c_gamma.acquire(1); o = c_out.acquire(1)
        k(x, g, o, ACT_SCALE, INV_ACT_SCALE)
        c_in.release(1); c_gamma.release(1); c_out.release(1)

    def w_q(c_act, c_w, c_out, k):
        a = c_act.acquire(1)
        o = c_out.acquire(1)
        for t in range_(N_TILES_Q):
            w = c_w.acquire(1)
            k(a, w, o, _i32(t))
            c_w.release(1)
        c_act.release(1); c_out.release(1)

    def w_o(c_act, c_w, c_out, k):
        a = c_act.acquire(1)
        o = c_out.acquire(1)
        for t in range_(N_TILES_O):
            w = c_w.acquire(1)
            k(a, w, o, _i32(t))
            c_w.release(1)
        c_act.release(1); c_out.release(1)

    def w_rope(c_x, c_cs, c_out, k):
        x = c_x.acquire(1); cs = c_cs.acquire(1); o = c_out.acquire(1)
        k(x, cs, o)
        c_x.release(1); c_cs.release(1); c_out.release(1)

    def w_qk(c_q, c_k, c_probs, k):
        q = c_q.acquire(1); kk = c_k.acquire(1); p = c_probs.acquire(1)
        k(q, kk, p)
        c_q.release(1); c_k.release(1); c_probs.release(1)

    def w_sv(c_v, c_probs, c_out, k):
        v = c_v.acquire(1); p = c_probs.acquire(1); o = c_out.acquire(1)
        k(v, p, o)
        c_v.release(1); c_probs.release(1); c_out.release(1)

    def w_add(c_a, c_b, c_out, k):
        a = c_a.acquire(1); b = c_b.acquire(1); o = c_out.acquire(1)
        k(a, b, o)
        c_a.release(1); c_b.release(1); c_out.release(1)

    PERCHAN_STACK = 8192
    workers = [
        Worker(w_rms, [of_xin.cons(), of_gamma.cons(), of_h1.prod(), k_rms],
               tile=Tile(5, 4)),
        Worker(w_q,
               [of_h1.cons(), of_wq.cons(), of_qf.prod(), k_q],
               tile=Tile(0, 2), stack_size=PERCHAN_STACK),
        Worker(w_rope,
               [of_qf.cons(), of_cs.cons(), of_qr.prod(), k_rope],
               tile=Tile(4, 4)),
        # flowkv_qk needs 16 KB stack for scores[T] + qvals[T] at T=128 (Bug 5).
        Worker(w_qk,
               [of_qr.cons(), of_kcache.cons(), of_probs.prod(), k_qk],
               tile=Tile(0, 4),
               stack_size=16384),
        Worker(w_sv,
               [of_vcache.cons(), of_probs.cons(), of_af.prod(), k_sv],
               tile=Tile(0, 5)),
        Worker(w_o,
               [of_af.cons(), of_wo.cons(), of_op.prod(), k_o],
               tile=Tile(3, 2), stack_size=PERCHAN_STACK),
        Worker(w_add,
               [of_op.cons(), of_xin.cons(), of_out.prod(), k_add],
               tile=Tile(6, 5)),
    ]

    rt = Runtime()
    with rt.sequence(rt_xin_ty, rt_w_ty, rt_kv_ty, rt_out_ty) as (
        xin, wblob, kvblob, out
    ):
        rt.start(*workers)

        xin_tg = rt.task_group()
        rt.fill(of_xin.prod(), xin, task_group=xin_tg)

        gam_tg = rt.task_group()
        rt.fill(of_gamma.prod(), wblob,
                tap=strided_tap(WEIGHTS_BYTES, OFF_GAMMA, GAMMA_BYTES,
                                GAMMA_BYTES, 1),
                task_group=gam_tg)

        wq_tg = rt.task_group()
        rt.fill(of_wq.prod(), wblob,
                tap=strided_tap(WEIGHTS_BYTES, OFF_WQ, WQ_SLOT, WQ_SLOT, N_TILES_Q),
                task_group=wq_tg)

        cs_tg = rt.task_group()
        rt.fill(of_cs.prod(), wblob,
                tap=strided_tap(WEIGHTS_BYTES, OFF_CS, CS_BYTES, CS_BYTES, 1),
                task_group=cs_tg)

        wo_tg = rt.task_group()
        rt.fill(of_wo.prod(), wblob,
                tap=strided_tap(WEIGHTS_BYTES, OFF_WO, WO_SLOT, WO_SLOT, N_TILES_O),
                task_group=wo_tg)

        kcache_tg = rt.task_group()
        rt.fill(of_kcache.prod(), kvblob,
                tap=strided_tap(KV_BYTES, OFF_K, KCACHE_PADDED,
                                KCACHE_PADDED, 1),
                task_group=kcache_tg)

        vcache_tg = rt.task_group()
        rt.fill(of_vcache.prod(), kvblob,
                tap=strided_tap(KV_BYTES, OFF_V, VCACHE_PADDED,
                                VCACHE_PADDED, 1),
                task_group=vcache_tg)

        out_tg = rt.task_group()
        rt.drain(of_out.cons(), out, wait=True, task_group=out_tg)

        for tg in [xin_tg, gam_tg, wq_tg, cs_tg, wo_tg, kcache_tg, vcache_tg, out_tg]:
            rt.finish_task_group(tg)

    return Program(NPU2(), rt).resolve_program()


def main():
    argparse.ArgumentParser().parse_args(sys.argv[1:])
    print(build())


if __name__ == "__main__":
    main()
