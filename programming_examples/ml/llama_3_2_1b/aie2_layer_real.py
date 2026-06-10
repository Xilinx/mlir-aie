"""Phase 3: full single-decoder-layer integration with all REAL kernels.

Mirrors aie2_layer.py's 16-worker topology but each kernel is now the
real (bit-exact) implementation from Phase 2. 4 consolidated runtime
buffers feed all worker fifos via TensorAccessPattern taps:

  arg0: x_in        (D int8)
  arg1: weights     (one packed blob; layout below)
  arg2: kv_cache    (K_cache || V_cache int8)
  arg3: layer_out   (D int8)

weights packed layout (offsets in bytes, all little-endian raw):

  off_gamma_in   : D * 2  (bf16)
  off_gamma_post : D * 2  (bf16)
  off_wq_packed  : QD*D + QD*4   (int8 weights row-major + int32 bias)
  off_wk_packed  : KVD*D + KVD*4
  off_wv_packed  : KVD*D + KVD*4
  off_wo_packed  : D*QD + D*4
  off_wg_packed  : HD*D + HD*4
  off_wu_packed  : HD*D + HD*4
  off_wd_packed  : D*HD + D*4
  off_cs_packed  : head_dim*2 * 2  (cos || sin, bf16)

Test side (test_layer_real.py) computes this layout, fills weights and
kv_cache with random values, runs the kernel, and compares bit-exact
against a numpy reference that chains the per-kernel references.

Sizes are tiny for first bring-up:
  D=64, QD=64, KVD=64, HD=256, head_dim=64, n_heads=1, n_kv=1, T=16

(Llama 3.2 1B values are D=2048, QD=2048, KVD=512, HD=8192.)
"""

import argparse
import sys

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile
from aie.helpers.taplib import TensorAccessPattern

# Sizes (small first-test). MUST match test_layer_real.py.
D = 64
QD = 64
KVD = 64
HD = 256
HEAD_D = 64
N_HEADS = 1
N_KV = 1
T = 16

# Bias entries are int32 = 4 bytes each.
WQ_BYTES = QD * D + QD * 4
WK_BYTES = KVD * D + KVD * 4
WV_BYTES = KVD * D + KVD * 4
WO_BYTES = D * QD + D * 4
WG_BYTES = HD * D + HD * 4
WU_BYTES = HD * D + HD * 4
WD_BYTES = D * HD + D * 4
GAMMA_BYTES = D * 2  # bf16
CS_BYTES = HEAD_D * 2 * 2  # cos || sin, bf16
KCACHE_BYTES = T * KVD  # int8
VCACHE_BYTES = T * KVD  # int8

OFF_GAMMA_IN = 0
OFF_GAMMA_POST = OFF_GAMMA_IN + GAMMA_BYTES
OFF_WQ = OFF_GAMMA_POST + GAMMA_BYTES
OFF_WK = OFF_WQ + WQ_BYTES
OFF_WV = OFF_WK + WK_BYTES
OFF_WO = OFF_WV + WV_BYTES
OFF_WG = OFF_WO + WO_BYTES
OFF_WU = OFF_WG + WG_BYTES
OFF_WD = OFF_WU + WU_BYTES
OFF_CS = OFF_WD + WD_BYTES
TOTAL_W = OFF_CS + CS_BYTES

KV_TOTAL = KCACHE_BYTES + VCACHE_BYTES  # K then V
OFF_K = 0
OFF_V = KCACHE_BYTES

# Per-call scales / shifts. Use uniform values for first integration.
# (Each layer in production would have its own calibrated scales.)
RIGHT_SHIFT = 12  # all gemm calls
ACT_SCALE = 0.05  # used by rmsnorm, rope as in/out scale
INV_ACT_SCALE = 1.0 / ACT_SCALE
UP_SCALE = 0.05
INV_OUT_SCALE_SI = 1.0 / 0.05
Q_SCALE = 0.05
K_SCALE = 0.05
V_SCALE = 0.05
INV_OUT_SCALE_AT = 1.0 / 0.05


def _i8(n):
    return np.ndarray[(int(n),), np.dtype[np.int8]]


def _bf16(n):
    from ml_dtypes import bfloat16

    return np.ndarray[(int(n),), np.dtype[bfloat16]]


def build():
    from ml_dtypes import bfloat16

    # Runtime-arg types.
    rt_xin_ty = _i8(D)
    rt_w_ty = _i8(TOTAL_W)
    rt_kv_ty = _i8(KV_TOTAL)
    rt_out_ty = _i8(D)

    # Per-fifo types.
    t_D_i8 = _i8(D)
    t_QD_i8 = _i8(QD)
    t_KVD_i8 = _i8(KVD)
    t_HD_i8 = _i8(HD)
    t_D_bf16 = _bf16(D)
    t_CS_bf16 = _bf16(HEAD_D * 2)  # cos || sin packed
    t_WQ_i8 = _i8(WQ_BYTES)
    t_WK_i8 = _i8(WK_BYTES)
    t_WV_i8 = _i8(WV_BYTES)
    t_WO_i8 = _i8(WO_BYTES)
    t_WG_i8 = _i8(WG_BYTES)
    t_WU_i8 = _i8(WU_BYTES)
    t_WD_i8 = _i8(WD_BYTES)
    t_KCACHE_i8 = _i8(KCACHE_BYTES)
    t_VCACHE_i8 = _i8(VCACHE_BYTES)
    t_PROBS_fp32 = np.ndarray[(T,), np.dtype[np.float32]]

    # --- ObjectFifos (one per dataflow edge). ---
    # Activation path. v0 drops k_proj/v_proj/rope_k -- the current
    # token's K/V would normally be roped (K only) and appended to the
    # cache; we pre-fill the whole cache from runtime args instead.
    of_xin = ObjectFifo(t_D_i8, depth=4, name="x_in")  # residual hold #1
    of_h1 = ObjectFifo(t_D_i8, name="h1")
    of_qf = ObjectFifo(t_QD_i8, name="qf")
    of_qr = ObjectFifo(t_QD_i8, name="qr")
    of_probs = ObjectFifo(t_PROBS_fp32, name="probs")
    of_af = ObjectFifo(t_QD_i8, name="af")
    of_op = ObjectFifo(t_D_i8, name="op")
    of_x1 = ObjectFifo(t_D_i8, depth=4, name="x1")  # residual hold #2
    of_h2 = ObjectFifo(t_D_i8, name="h2")
    of_gf = ObjectFifo(t_HD_i8, name="gf")
    of_uf = ObjectFifo(t_HD_i8, name="uf")
    of_sf = ObjectFifo(t_HD_i8, name="sf")
    of_df = ObjectFifo(t_D_i8, name="df")
    of_out = ObjectFifo(t_D_i8, name="layer_out")

    # Weight streams (one per consumer kernel; sourced from weights blob via tap).
    of_gam_in = ObjectFifo(t_D_bf16, name="gam_in")
    of_gam_post = ObjectFifo(t_D_bf16, name="gam_post")
    of_wq = ObjectFifo(t_WQ_i8, name="wq")
    of_wo = ObjectFifo(t_WO_i8, name="wo")
    of_wg = ObjectFifo(t_WG_i8, name="wg")
    of_wu = ObjectFifo(t_WU_i8, name="wu")
    of_wd = ObjectFifo(t_WD_i8, name="wd")
    of_cs = ObjectFifo(t_CS_bf16, name="cs")
    of_kcache = ObjectFifo(t_KCACHE_i8, name="kcache")
    of_vcache = ObjectFifo(t_VCACHE_i8, name="vcache")

    # --- Kernels ---
    KO_RMS = "llama_rmsnorm_int8.cc.o"
    KO_GEMM = "llama_gemm_int8_srs.cc.o"
    KO_ROPE = "llama_rope_int8.cc.o"
    KO_SILU = "llama_silu_mul_int8.cc.o"
    KO_FKV = "llama_flowkv.cc.o"

    k_rms_D = Kernel(
        "llama_rmsnorm_int8", KO_RMS, [t_D_i8, t_D_bf16, t_D_i8, np.float32, np.float32]
    )

    # gemm signature is shape-specific (act and out sizes vary), need
    # per-call-site Kernel objects with the right types but SAME C symbol.
    # Since IRON requires unique C symbols per Kernel, we need per-shape
    # symbols -- but the kernel impl is parameterized by LLAMA_GEMM_K/N
    # via #defines. For this integration test we use ONE shape for all
    # gemms (K=N=D=64, ignoring the HD asymmetry). FFN gates/up/down at
    # HD=256 require a different #define-shape kernel; for first test
    # we drop FFN to single-shape too (HD=D). This is a sizing
    # compromise for v0; production uses N-shape-specific .o files.
    k_gemm_DD = Kernel(
        "llama_gemm_int8_srs_D_to_D", KO_GEMM, [t_D_i8, t_WQ_i8, t_D_i8, np.int32]
    )
    k_gemm_DHD = Kernel(
        "llama_gemm_int8_srs_D_to_HD", KO_GEMM, [t_D_i8, t_WG_i8, t_HD_i8, np.int32]
    )
    k_gemm_HDD = Kernel(
        "llama_gemm_int8_srs_HD_to_D", KO_GEMM, [t_HD_i8, t_WD_i8, t_D_i8, np.int32]
    )

    k_rope_i8 = Kernel(
        "llama_rope_int8", KO_ROPE, [t_QD_i8, t_CS_bf16, t_QD_i8, np.float32]
    )

    k_silu_HD = Kernel(
        "llama_silu_mul_int8",
        KO_SILU,
        [t_HD_i8, t_HD_i8, t_HD_i8, np.float32, np.float32],
    )

    k_fkv_qk = Kernel(
        "llama_flowkv_qk",
        KO_FKV,
        [t_QD_i8, t_KCACHE_i8, t_PROBS_fp32, np.float32, np.float32],
    )
    k_fkv_sv = Kernel(
        "llama_flowkv_sv",
        KO_FKV,
        [t_VCACHE_i8, t_PROBS_fp32, t_QD_i8, np.float32, np.float32],
    )

    # Residual add is a separate kernel; reuse the layer_pt add stub
    # since the math is the same (int8 wrap add). Could replace with a
    # real "llama_add_int8" kernel later -- the stub IS correct.
    k_add_D = Kernel("llama_pt_add_D", "llama_layer_pt.cc.o", [t_D_i8, t_D_i8, t_D_i8])

    # --- Worker bodies ---
    def w_rms(c_in, c_gamma, c_out, k):
        x = c_in.acquire(1)
        g = c_gamma.acquire(1)
        o = c_out.acquire(1)
        k(x, g, o, ACT_SCALE, INV_ACT_SCALE)
        c_in.release(1)
        c_gamma.release(1)
        c_out.release(1)

    def w_gemm(c_act, c_w, c_out, k):
        a = c_act.acquire(1)
        w = c_w.acquire(1)
        o = c_out.acquire(1)
        k(a, w, o, RIGHT_SHIFT)
        c_act.release(1)
        c_w.release(1)
        c_out.release(1)

    def w_rope(c_x, c_cs, c_out, k):
        x = c_x.acquire(1)
        cs = c_cs.acquire(1)
        o = c_out.acquire(1)
        k(x, cs, o, ACT_SCALE)
        c_x.release(1)
        c_cs.release(1)
        c_out.release(1)

    def w_qk(c_q, c_k, c_probs, k):
        q = c_q.acquire(1)
        kk = c_k.acquire(1)
        p = c_probs.acquire(1)
        k(q, kk, p, Q_SCALE, K_SCALE)
        c_q.release(1)
        c_k.release(1)
        c_probs.release(1)

    def w_sv(c_v, c_probs, c_out, k):
        v = c_v.acquire(1)
        p = c_probs.acquire(1)
        o = c_out.acquire(1)
        k(v, p, o, V_SCALE, INV_OUT_SCALE_AT)
        c_v.release(1)
        c_probs.release(1)
        c_out.release(1)

    def w_silu(c_g, c_u, c_out, k):
        g = c_g.acquire(1)
        u = c_u.acquire(1)
        o = c_out.acquire(1)
        k(g, u, o, UP_SCALE, INV_OUT_SCALE_SI)
        c_g.release(1)
        c_u.release(1)
        c_out.release(1)

    def w_add(c_a, c_b, c_out, k):
        a = c_a.acquire(1)
        b = c_b.acquire(1)
        o = c_out.acquire(1)
        k(a, b, o)
        c_a.release(1)
        c_b.release(1)
        c_out.release(1)

    # --- Workers (in dataflow order, pinned to DECODE_PLACEMENT-ish tiles) ---
    workers = [
        # rmsnorm1
        Worker(
            w_rms,
            [of_xin.cons(), of_gam_in.cons(), of_h1.prod(), k_rms_D],
            tile=Tile(5, 4),
        ),
        # q_proj
        Worker(
            w_gemm,
            [of_h1.cons(), of_wq.cons(), of_qf.prod(), k_gemm_DD],
            tile=Tile(0, 2),
        ),
        # rope_q
        Worker(
            w_rope,
            [of_qf.cons(), of_cs.cons(), of_qr.prod(), k_rope_i8],
            tile=Tile(4, 4),
        ),
        # flowkv_qk
        Worker(
            w_qk,
            [of_qr.cons(), of_kcache.cons(), of_probs.prod(), k_fkv_qk],
            tile=Tile(0, 4),
        ),
        # flowkv_sv
        Worker(
            w_sv,
            [of_vcache.cons(), of_probs.cons(), of_af.prod(), k_fkv_sv],
            tile=Tile(0, 5),
        ),
        # o_proj
        Worker(
            w_gemm,
            [of_af.cons(), of_wo.cons(), of_op.prod(), k_gemm_DD],
            tile=Tile(3, 2),
        ),
        # add1
        Worker(
            w_add, [of_op.cons(), of_xin.cons(), of_x1.prod(), k_add_D], tile=Tile(6, 5)
        ),
        # rmsnorm2
        Worker(
            w_rms,
            [of_x1.cons(), of_gam_post.cons(), of_h2.prod(), k_rms_D],
            tile=Tile(6, 4),
        ),
        # gate/up/silu/down: HD=256 path. For first integration we keep
        # HD=D so we reuse k_gemm_DD; tiling-up via HD>D is a follow-up
        # that needs a separate-shape gemm .o.
        Worker(
            w_gemm,
            [of_h2.cons(), of_wg.cons(), of_gf.prod(), k_gemm_DHD],
            tile=Tile(4, 2),
        ),
        Worker(
            w_gemm,
            [of_h2.cons(), of_wu.cons(), of_uf.prod(), k_gemm_DHD],
            tile=Tile(5, 2),
        ),
        Worker(
            w_silu,
            [of_gf.cons(), of_uf.cons(), of_sf.prod(), k_silu_HD],
            tile=Tile(4, 5),
        ),
        Worker(
            w_gemm,
            [of_sf.cons(), of_wd.cons(), of_df.prod(), k_gemm_HDD],
            tile=Tile(6, 2),
        ),
        # add2
        Worker(
            w_add, [of_df.cons(), of_x1.cons(), of_out.prod(), k_add_D], tile=Tile(7, 5)
        ),
    ]

    rt = Runtime()
    with rt.sequence(rt_xin_ty, rt_w_ty, rt_kv_ty, rt_out_ty) as (xin, wblob, kv, out):
        rt.start(*workers)

        rt.fill(of_xin.prod(), xin)

        def tap_w(offset, nbytes):
            return TensorAccessPattern(
                tensor_dims=[TOTAL_W],
                offset=offset,
                sizes=[1, 1, 1, nbytes],
                strides=[0, 0, 0, 1],
            )

        rt.fill(of_gam_in.prod(), wblob, tap=tap_w(OFF_GAMMA_IN, GAMMA_BYTES))
        rt.fill(of_gam_post.prod(), wblob, tap=tap_w(OFF_GAMMA_POST, GAMMA_BYTES))
        rt.fill(of_wq.prod(), wblob, tap=tap_w(OFF_WQ, WQ_BYTES))
        rt.fill(of_wo.prod(), wblob, tap=tap_w(OFF_WO, WO_BYTES))
        rt.fill(of_wg.prod(), wblob, tap=tap_w(OFF_WG, WG_BYTES))
        rt.fill(of_wu.prod(), wblob, tap=tap_w(OFF_WU, WU_BYTES))
        rt.fill(of_wd.prod(), wblob, tap=tap_w(OFF_WD, WD_BYTES))
        rt.fill(of_cs.prod(), wblob, tap=tap_w(OFF_CS, CS_BYTES))

        def tap_kv(offset, nbytes):
            return TensorAccessPattern(
                tensor_dims=[KV_TOTAL],
                offset=offset,
                sizes=[1, 1, 1, nbytes],
                strides=[0, 0, 0, 1],
            )

        rt.fill(of_kcache.prod(), kv, tap=tap_kv(OFF_K, KCACHE_BYTES))
        rt.fill(of_vcache.prod(), kv, tap=tap_kv(OFF_V, VCACHE_BYTES))

        rt.drain(of_out.cons(), out, wait=True)

    return Program(NPU2(), rt).resolve_program()


def main():
    argparse.ArgumentParser().parse_args(sys.argv[1:])
    print(build())


if __name__ == "__main__":
    main()
