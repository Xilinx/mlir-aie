"""Phase 6b: N-layer decode chain + lm_head GEMM + on-device sampler.

Builds on aie2_chain_real.py: same residual-loop chain, but the last
add2's output flows into an `lm_head` GEMM (D->V int8) instead of being
drained, and those logits feed the Phase-6a sample kernel. The host
drains a single int32 token id.

Runtime sequence (in order):
   xin     (int8[D])        # initial activation
   wblob   (int8[TOTAL_W])  # N per-layer weight blobs (chain)
   kvblob  (int8[TOTAL_KV]) # N per-layer KV-cache blobs (chain)
   lm_w    (int8[LMW])      # lm_head packed weights + biases
   params  (uint32[3])      # [temperature_bits, top_k, seed]
   token   (int32[1])       # output token id

V is chosen so the lm_head weight buffer at depth=2 fits compute-tile
L1 alongside its act/out buffers (same envelope as the chain's other
projections). Real Llama V=128256 requires tiling -- 6c/7 work.
"""

import argparse
import os as _os
import sys

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern

# Per-layer chain sizes (must match aie2_chain_real.py).
D = 64
QD = 64
KVD = 64
HD = 256
HEAD_D = 64
N_HEADS = 1
N_KV = 1
T = 16
N_LAYERS = int(_os.environ.get("LLAMA_CHAIN_N", "16"))

# lm_head vocab. Must match -DLLAMA_GEMM_VOCAB / -DLLAMA_SAMPLE_VOCAB at
# kernel compile time. Defaults to 256 (fits L1 at depth=2 with margin).
V = int(_os.environ.get("LLAMA_CHAIN_SAMPLE_V", "256"))

# Per-layer byte sizes (same as chain_real).
WQ_BYTES = QD * D + QD * 4
WO_BYTES = D * QD + D * 4
WG_BYTES = HD * D + HD * 4
WU_BYTES = HD * D + HD * 4
WD_BYTES = D * HD + D * 4
GAMMA_BYTES = D * 2
CS_BYTES = HEAD_D * 2 * 2
KCACHE_BYTES = T * KVD
VCACHE_BYTES = T * KVD

# Per-layer weights blob layout.
OFF_GAMMA_IN = 0
OFF_GAMMA_POST = OFF_GAMMA_IN + GAMMA_BYTES
OFF_WQ = OFF_GAMMA_POST + GAMMA_BYTES
OFF_WO = OFF_WQ + WQ_BYTES
OFF_WG = OFF_WO + WO_BYTES
OFF_WU = OFF_WG + WG_BYTES
OFF_WD = OFF_WU + WU_BYTES
OFF_CS = OFF_WD + WD_BYTES
PER_LAYER_W = OFF_CS + CS_BYTES

# KV cache blob layout.
OFF_K = 0
OFF_V = OFF_K + KCACHE_BYTES
PER_LAYER_KV = OFF_V + VCACHE_BYTES

# Total blob sizes.
TOTAL_W = N_LAYERS * PER_LAYER_W
TOTAL_KV = N_LAYERS * PER_LAYER_KV

# lm_head matrix layout: [V*D int8 weights | V*4 int32 bias].
LMW_BYTES = V * D + V * 4

# Params layout (3 uint32 = 12 bytes): [temperature_bits, top_k, seed].
PARAMS_BYTES = 12

# Consolidated "auxiliary" blob = lm_head + params + ... so that the
# whole xclbin fits inside DefaultNPURuntime's ~5-arg ceiling. Layout:
#   [0,       LMW_BYTES)               : lm_head matrix
#   [LMW_BYTES, LMW_BYTES + PARAMS)    : sampler params
AUX_OFF_LMW = 0
AUX_OFF_PARAMS = AUX_OFF_LMW + LMW_BYTES
AUX_BYTES = AUX_OFF_PARAMS + PARAMS_BYTES

# Per-call scales (uniform across layers for first integration test).
RIGHT_SHIFT = 12
ACT_SCALE = 0.05
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

    rt_xin_ty = _i8(D)
    rt_w_ty = _i8(TOTAL_W)
    rt_kv_ty = _i8(TOTAL_KV)
    rt_aux_ty = _i8(AUX_BYTES)  # lm_head + params concat
    rt_token_ty = np.ndarray[(1,), np.dtype[np.int32]]

    t_D_i8 = _i8(D)
    t_QD_i8 = _i8(QD)
    t_HD_i8 = _i8(HD)
    t_V_i8 = _i8(V)
    t_LMW_i8 = _i8(LMW_BYTES)
    t_D_bf16 = _bf16(D)
    t_CS_bf16 = _bf16(HEAD_D * 2)
    t_WQ_i8 = _i8(WQ_BYTES)
    t_WO_i8 = _i8(WO_BYTES)
    t_WG_i8 = _i8(WG_BYTES)
    t_WU_i8 = _i8(WU_BYTES)
    t_WD_i8 = _i8(WD_BYTES)
    t_KCACHE_i8 = _i8(KCACHE_BYTES)
    t_VCACHE_i8 = _i8(VCACHE_BYTES)
    t_PROBS_fp32 = np.ndarray[(T,), np.dtype[np.float32]]
    t_PARAMS_u32 = np.ndarray[(3,), np.dtype[np.uint32]]
    t_TOK_i32 = np.ndarray[(1,), np.dtype[np.int32]]

    # --- Chain fifos (same as chain_real, depths at 2) ---
    of_seed = ObjectFifo(t_D_i8, depth=1, name="seed")
    of_back = ObjectFifo(t_D_i8, depth=2, name="back")
    of_routed = ObjectFifo(t_D_i8, depth=2, name="routed")

    # The "final-x" fifo replaces chain_real's of_out: last add2 writes
    # here, lm_head reads here. depth=1 because lm_head consumes exactly
    # one slot per xclbin dispatch.
    of_x_final = ObjectFifo(t_D_i8, depth=1, name="x_final")

    of_h1 = ObjectFifo(t_D_i8, name="h1")
    of_qf = ObjectFifo(t_QD_i8, name="qf")
    of_qr = ObjectFifo(t_QD_i8, name="qr")
    of_probs = ObjectFifo(t_PROBS_fp32, name="probs")
    of_af = ObjectFifo(t_QD_i8, name="af")
    of_op = ObjectFifo(t_D_i8, name="op")
    of_x1 = ObjectFifo(t_D_i8, depth=2, name="x1")
    of_h2 = ObjectFifo(t_D_i8, name="h2")
    of_gf = ObjectFifo(t_HD_i8, name="gf")
    of_uf = ObjectFifo(t_HD_i8, name="uf")
    of_sf = ObjectFifo(t_HD_i8, name="sf")
    of_df = ObjectFifo(t_D_i8, name="df")

    of_gam_in = ObjectFifo(t_D_bf16, depth=2, name="gam_in")
    of_gam_post = ObjectFifo(t_D_bf16, depth=2, name="gam_post")
    of_wq = ObjectFifo(t_WQ_i8, depth=2, name="wq")
    of_wo = ObjectFifo(t_WO_i8, depth=2, name="wo")
    of_wg = ObjectFifo(t_WG_i8, depth=2, name="wg")
    of_wu = ObjectFifo(t_WU_i8, depth=2, name="wu")
    of_wd = ObjectFifo(t_WD_i8, depth=2, name="wd")
    of_cs = ObjectFifo(t_CS_bf16, depth=2, name="cs")
    of_kcache = ObjectFifo(t_KCACHE_i8, depth=2, name="kcache")
    of_vcache = ObjectFifo(t_VCACHE_i8, depth=2, name="vcache")

    # --- lm_head + sample fifos ---
    of_lm_w = ObjectFifo(t_LMW_i8, depth=2, name="lm_w")
    of_logits = ObjectFifo(t_V_i8, depth=2, name="logits")
    of_params = ObjectFifo(t_PARAMS_u32, depth=1, name="params")
    of_token = ObjectFifo(t_TOK_i32, depth=1, name="token")

    KO_RMS = "llama_rmsnorm_int8.cc.o"
    KO_GEMM = "llama_gemm_int8_srs.cc.o"
    KO_ROPE = "llama_rope_int8.cc.o"
    KO_SILU = "llama_silu_mul_int8.cc.o"
    KO_FKV = "llama_flowkv.cc.o"
    KO_PT = "llama_layer_pt.cc.o"
    KO_SMP = "llama_sample.cc.o"

    k_rms = Kernel(
        "llama_rmsnorm_int8", KO_RMS, [t_D_i8, t_D_bf16, t_D_i8, np.float32, np.float32]
    )
    k_gemm_DD = Kernel(
        "llama_gemm_int8_srs_D_to_D", KO_GEMM, [t_D_i8, t_WQ_i8, t_D_i8, np.int32]
    )
    k_gemm_DHD = Kernel(
        "llama_gemm_int8_srs_D_to_HD", KO_GEMM, [t_D_i8, t_WG_i8, t_HD_i8, np.int32]
    )
    k_gemm_HDD = Kernel(
        "llama_gemm_int8_srs_HD_to_D", KO_GEMM, [t_HD_i8, t_WD_i8, t_D_i8, np.int32]
    )
    k_gemm_DV = Kernel(
        "llama_gemm_int8_srs_D_to_V", KO_GEMM, [t_D_i8, t_LMW_i8, t_V_i8, np.int32]
    )
    k_rope = Kernel(
        "llama_rope_int8", KO_ROPE, [t_QD_i8, t_CS_bf16, t_QD_i8, np.float32]
    )
    k_silu = Kernel(
        "llama_silu_mul_int8",
        KO_SILU,
        [t_HD_i8, t_HD_i8, t_HD_i8, np.float32, np.float32],
    )
    k_qk = Kernel(
        "llama_flowkv_qk",
        KO_FKV,
        [t_QD_i8, t_KCACHE_i8, t_PROBS_fp32, np.float32, np.float32],
    )
    k_sv = Kernel(
        "llama_flowkv_sv",
        KO_FKV,
        [t_VCACHE_i8, t_PROBS_fp32, t_QD_i8, np.float32, np.float32],
    )
    k_add = Kernel("llama_pt_add_D", KO_PT, [t_D_i8, t_D_i8, t_D_i8])
    k_sample = Kernel("llama_sample", KO_SMP, [t_V_i8, t_TOK_i32, t_PARAMS_u32])

    # --- Chain worker bodies (loop N_LAYERS times) ---
    def w_router(c_seed, c_back, p_routed):
        x = c_seed.acquire(1)
        o = p_routed.acquire(1)
        for i in range(D):
            o[i] = x[i]
        p_routed.release(1)
        c_seed.release(1)
        for _ in range_(N_LAYERS - 1):
            x = c_back.acquire(1)
            o = p_routed.acquire(1)
            for i in range(D):
                o[i] = x[i]
            p_routed.release(1)
            c_back.release(1)

    def w_rms(c_in, c_gamma, c_out, k):
        for _ in range_(N_LAYERS):
            x = c_in.acquire(1)
            g = c_gamma.acquire(1)
            o = c_out.acquire(1)
            k(x, g, o, ACT_SCALE, INV_ACT_SCALE)
            c_in.release(1)
            c_gamma.release(1)
            c_out.release(1)

    def w_gemm(c_act, c_w, c_out, k):
        for _ in range_(N_LAYERS):
            a = c_act.acquire(1)
            w = c_w.acquire(1)
            o = c_out.acquire(1)
            k(a, w, o, RIGHT_SHIFT)
            c_act.release(1)
            c_w.release(1)
            c_out.release(1)

    def w_rope_fn(c_x, c_cs, c_out, k):
        for _ in range_(N_LAYERS):
            x = c_x.acquire(1)
            cs = c_cs.acquire(1)
            o = c_out.acquire(1)
            k(x, cs, o, ACT_SCALE)
            c_x.release(1)
            c_cs.release(1)
            c_out.release(1)

    def w_qk_fn(c_q, c_k, c_probs, k):
        for _ in range_(N_LAYERS):
            q = c_q.acquire(1)
            kk = c_k.acquire(1)
            p = c_probs.acquire(1)
            k(q, kk, p, Q_SCALE, K_SCALE)
            c_q.release(1)
            c_k.release(1)
            c_probs.release(1)

    def w_sv_fn(c_v, c_probs, c_out, k):
        for _ in range_(N_LAYERS):
            v = c_v.acquire(1)
            p = c_probs.acquire(1)
            o = c_out.acquire(1)
            k(v, p, o, V_SCALE, INV_OUT_SCALE_AT)
            c_v.release(1)
            c_probs.release(1)
            c_out.release(1)

    def w_silu_fn(c_g, c_u, c_out, k):
        for _ in range_(N_LAYERS):
            g = c_g.acquire(1)
            u = c_u.acquire(1)
            o = c_out.acquire(1)
            k(g, u, o, UP_SCALE, INV_OUT_SCALE_SI)
            c_g.release(1)
            c_u.release(1)
            c_out.release(1)

    def w_add_fn(c_a, c_b, c_out, k):
        for _ in range_(N_LAYERS):
            a = c_a.acquire(1)
            b = c_b.acquire(1)
            o = c_out.acquire(1)
            k(a, b, o)
            c_a.release(1)
            c_b.release(1)
            c_out.release(1)

    # add2: writes to `back` for layers 0..N-2, to `x_final` on layer N-1.
    def w_add2_tail(c_a, c_b, c_back, c_xfinal, k):
        for _ in range_(N_LAYERS - 1):
            a = c_a.acquire(1)
            b = c_b.acquire(1)
            o = c_back.acquire(1)
            k(a, b, o)
            c_a.release(1)
            c_b.release(1)
            c_back.release(1)
        a = c_a.acquire(1)
        b = c_b.acquire(1)
        o = c_xfinal.acquire(1)
        k(a, b, o)
        c_a.release(1)
        c_b.release(1)
        c_xfinal.release(1)

    # --- lm_head + sample workers (fire once, no range_) ---
    def w_lm_head(c_x, c_w, c_logits, k):
        x = c_x.acquire(1)
        w = c_w.acquire(1)
        o = c_logits.acquire(1)
        k(x, w, o, RIGHT_SHIFT)
        c_x.release(1)
        c_w.release(1)
        c_logits.release(1)

    def w_sample(c_logits, c_params, c_token, k):
        l = c_logits.acquire(1)
        p = c_params.acquire(1)
        t = c_token.acquire(1)
        k(l, t, p)
        c_logits.release(1)
        c_params.release(1)
        c_token.release(1)

    # --- Workers ---
    workers = [
        Worker(
            w_router,
            [of_seed.cons(), of_back.cons(), of_routed.prod()],
            tile=Tile(7, 4),
        ),
        Worker(
            w_rms,
            [of_routed.cons(), of_gam_in.cons(), of_h1.prod(), k_rms],
            tile=Tile(5, 4),
        ),
        Worker(
            w_gemm,
            [of_h1.cons(), of_wq.cons(), of_qf.prod(), k_gemm_DD],
            tile=Tile(0, 2),
        ),
        Worker(
            w_rope_fn,
            [of_qf.cons(), of_cs.cons(), of_qr.prod(), k_rope],
            tile=Tile(4, 4),
        ),
        Worker(
            w_qk_fn,
            [of_qr.cons(), of_kcache.cons(), of_probs.prod(), k_qk],
            tile=Tile(0, 4),
        ),
        Worker(
            w_sv_fn,
            [of_vcache.cons(), of_probs.cons(), of_af.prod(), k_sv],
            tile=Tile(0, 5),
        ),
        Worker(
            w_gemm,
            [of_af.cons(), of_wo.cons(), of_op.prod(), k_gemm_DD],
            tile=Tile(3, 2),
        ),
        Worker(
            w_add_fn,
            [of_op.cons(), of_routed.cons(), of_x1.prod(), k_add],
            tile=Tile(6, 5),
        ),
        Worker(
            w_rms,
            [of_x1.cons(), of_gam_post.cons(), of_h2.prod(), k_rms],
            tile=Tile(6, 4),
        ),
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
            w_silu_fn,
            [of_gf.cons(), of_uf.cons(), of_sf.prod(), k_silu],
            tile=Tile(4, 5),
        ),
        Worker(
            w_gemm,
            [of_sf.cons(), of_wd.cons(), of_df.prod(), k_gemm_HDD],
            tile=Tile(6, 2),
        ),
        Worker(
            w_add2_tail,
            [of_df.cons(), of_x1.cons(), of_back.prod(), of_x_final.prod(), k_add],
            tile=Tile(7, 5),
        ),
        # lm_head: D->V GEMM, once per dispatch.
        Worker(
            w_lm_head,
            [of_x_final.cons(), of_lm_w.cons(), of_logits.prod(), k_gemm_DV],
            tile=Tile(7, 2),
        ),
        # sample: V int8 logits + 3xuint32 params -> 1 int32 token.
        Worker(
            w_sample,
            [of_logits.cons(), of_params.cons(), of_token.prod(), k_sample],
            tile=Tile(5, 5),
            stack_size=16384,
        ),  # Bug 5: kV-sized stack arrays
    ]

    rt = Runtime()
    with rt.sequence(rt_xin_ty, rt_w_ty, rt_kv_ty, rt_aux_ty, rt_token_ty) as (
        xin,
        wblob,
        kvblob,
        aux,
        token,
    ):
        rt.start(*workers)

        seed_tg = rt.task_group()
        rt.fill(of_seed.prod(), xin, task_group=seed_tg)

        def factor(nb):
            if nb <= 1023:
                return (1, nb)
            for inner in range(min(nb, 1023), 0, -1):
                if nb % inner == 0 and nb // inner <= 1023:
                    return (nb // inner, inner)
            raise ValueError(f"can't factor nbytes={nb} into two <=1023 dims")

        def factor_tap(nbytes):
            outer, inner = factor(nbytes)
            return [1, 1, outer, inner]

        def factor_strides(nbytes):
            outer, inner = factor(nbytes)
            return [0, 0, inner, 1]

        def tap_strided(total, base_off, per_layer_stride, nbytes, n):
            outer, inner = factor(nbytes)
            return TensorAccessPattern(
                tensor_dims=[total],
                offset=base_off,
                sizes=[1, n, outer, inner],
                strides=[0, per_layer_stride, inner, 1],
            )

        tgs = []

        def per_fifo(fifo_prod, src, off_per_layer, off_offset, nbytes, total):
            tg = rt.task_group()
            tgs.append(tg)
            rt.fill(
                fifo_prod,
                src,
                tap=tap_strided(total, off_offset, off_per_layer, nbytes, N_LAYERS),
                task_group=tg,
            )

        per_fifo(
            of_gam_in.prod(), wblob, PER_LAYER_W, OFF_GAMMA_IN, GAMMA_BYTES, TOTAL_W
        )
        per_fifo(
            of_gam_post.prod(), wblob, PER_LAYER_W, OFF_GAMMA_POST, GAMMA_BYTES, TOTAL_W
        )
        per_fifo(of_wq.prod(), wblob, PER_LAYER_W, OFF_WQ, WQ_BYTES, TOTAL_W)
        per_fifo(of_wo.prod(), wblob, PER_LAYER_W, OFF_WO, WO_BYTES, TOTAL_W)
        per_fifo(of_wg.prod(), wblob, PER_LAYER_W, OFF_WG, WG_BYTES, TOTAL_W)
        per_fifo(of_wu.prod(), wblob, PER_LAYER_W, OFF_WU, WU_BYTES, TOTAL_W)
        per_fifo(of_wd.prod(), wblob, PER_LAYER_W, OFF_WD, WD_BYTES, TOTAL_W)
        per_fifo(of_cs.prod(), wblob, PER_LAYER_W, OFF_CS, CS_BYTES, TOTAL_W)
        per_fifo(of_kcache.prod(), kvblob, PER_LAYER_KV, OFF_K, KCACHE_BYTES, TOTAL_KV)
        per_fifo(of_vcache.prod(), kvblob, PER_LAYER_KV, OFF_V, VCACHE_BYTES, TOTAL_KV)

        # lm_head matrix: pull from aux at offset 0, LMW_BYTES wide.
        lmw_tg = rt.task_group()
        rt.fill(
            of_lm_w.prod(),
            aux,
            tap=TensorAccessPattern(
                tensor_dims=[AUX_BYTES],
                offset=AUX_OFF_LMW,
                sizes=factor_tap(LMW_BYTES),
                strides=factor_strides(LMW_BYTES),
            ),
            task_group=lmw_tg,
        )

        # Sampler params: pull from aux at PARAMS offset, 12 bytes wide.
        params_tg = rt.task_group()
        rt.fill(
            of_params.prod(),
            aux,
            tap=TensorAccessPattern(
                tensor_dims=[AUX_BYTES],
                offset=AUX_OFF_PARAMS,
                sizes=[1, 1, 1, PARAMS_BYTES],
                strides=[0, 0, 0, 1],
            ),
            task_group=params_tg,
        )

        # Drain output token (int32[1]).
        out_tg = rt.task_group()
        rt.drain(of_token.cons(), token, wait=True, task_group=out_tg)

        rt.finish_task_group(seed_tg)
        for tg in tgs:
            rt.finish_task_group(tg)
        rt.finish_task_group(lmw_tg)
        rt.finish_task_group(params_tg)
        rt.finish_task_group(out_tg)

    return Program(NPU2(), rt).resolve_program()


def main():
    argparse.ArgumentParser().parse_args(sys.argv[1:])
    print(build())


if __name__ == "__main__":
    main()
