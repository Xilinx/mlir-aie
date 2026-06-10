"""Phase 4: N-layer decode chain with reused tiles + per-layer weight streaming.

Ports the `build_decode_design` pattern from cautious-eureka:
ONE set of workers, each looping N_LAYERS times. Per-layer weights
are delivered as N_LAYERS successive fills on each weight ObjectFifo.
A `router` worker handles the residual loop-back:

  - layer 0:        seed (from runtime)  -> rmsnorm1 + add1-residual
  - layers 1..n-1:  back  (from prev add2) -> rmsnorm1 + add1-residual
  - after loop:     back (from last add2) -> layer_out (to runtime)

Same per-kernel set as the single-layer aie2_layer_real.py (rmsnorm /
gemm DD/DHD/HDD / rope / flowkv / silu / add). Adds one router worker
on a spare tile.
"""

import argparse
import sys

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern

# Llama 3.2 1B chain integration sizes (small first-bring-up).
D = 64
QD = 64
KVD = 64
HD = 256
HEAD_D = 64
N_HEADS = 1
N_KV = 1
T = 16  # KV cache length
import os as _os

N_LAYERS = int(_os.environ.get("LLAMA_CHAIN_N", "16"))

# Per-layer byte sizes (must match test_chain_real.py).
WQ_BYTES = QD * D + QD * 4
WO_BYTES = D * QD + D * 4
WG_BYTES = HD * D + HD * 4
WU_BYTES = HD * D + HD * 4
WD_BYTES = D * HD + D * 4
GAMMA_BYTES = D * 2
CS_BYTES = HEAD_D * 2 * 2
KCACHE_BYTES = T * KVD
VCACHE_BYTES = T * KVD

# Per-layer weights blob layout (within one layer's slice).
OFF_GAMMA_IN = 0
OFF_GAMMA_POST = OFF_GAMMA_IN + GAMMA_BYTES
OFF_WQ = OFF_GAMMA_POST + GAMMA_BYTES
OFF_WO = OFF_WQ + WQ_BYTES
OFF_WG = OFF_WO + WO_BYTES
OFF_WU = OFF_WG + WG_BYTES
OFF_WD = OFF_WU + WU_BYTES
OFF_CS = OFF_WD + WD_BYTES
PER_LAYER_W = OFF_CS + CS_BYTES

# Per-layer KV cache blob layout.
OFF_K = 0
OFF_V = OFF_K + KCACHE_BYTES
PER_LAYER_KV = OFF_V + VCACHE_BYTES

# Total blobs over all layers.
TOTAL_W = N_LAYERS * PER_LAYER_W
TOTAL_KV = N_LAYERS * PER_LAYER_KV

# Per-call scales (uniform across layers for first test).
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
    rt_out_ty = _i8(D)

    t_D_i8 = _i8(D)
    t_QD_i8 = _i8(QD)
    t_HD_i8 = _i8(HD)
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

    # Residual chain: seed + back -> router -> routed (broadcast to
    # rmsnorm1 AND add1 via IRON's native multi-consumer fan-out);
    # add2 -> back. Single output from router avoids per-element data
    # copy that IRON buffers don't support natively as a slice store.
    # Depths: pingpong (2) everywhere. Earlier defensive depth=N_LAYERS
    # blew the 64KB compute-tile L1 budget at N>=5 (wg cons buff alone
    # = N*17408 bytes). With shim BD-chain (single dma task per fifo
    # walking N layers via TAP) backpressure handles per-layer sync.
    of_seed = ObjectFifo(t_D_i8, depth=1, name="seed")
    of_back = ObjectFifo(t_D_i8, depth=2, name="back")
    of_routed = ObjectFifo(
        t_D_i8, depth=2, name="routed"
    )  # fans out to rmsnorm1 + add1
    of_out = ObjectFifo(t_D_i8, depth=1, name="layer_out")

    of_h1 = ObjectFifo(t_D_i8, name="h1")
    of_qf = ObjectFifo(t_QD_i8, name="qf")
    of_qr = ObjectFifo(t_QD_i8, name="qr")
    of_probs = ObjectFifo(t_PROBS_fp32, name="probs")
    of_af = ObjectFifo(t_QD_i8, name="af")
    of_op = ObjectFifo(t_D_i8, name="op")
    of_x1 = ObjectFifo(t_D_i8, depth=2, name="x1")  # residual hold #2
    of_h2 = ObjectFifo(t_D_i8, name="h2")
    of_gf = ObjectFifo(t_HD_i8, name="gf")
    of_uf = ObjectFifo(t_HD_i8, name="uf")
    of_sf = ObjectFifo(t_HD_i8, name="sf")
    of_df = ObjectFifo(t_D_i8, name="df")

    # Per-kernel weight streams: pingpong (depth=2). Shim BD-chain
    # walks all N per-layer slices via TAP (see runtime section below),
    # and backpressure from the consumer naturally serializes the
    # cross-layer delivery. Earlier depth=N_LAYERS blew compute tile L1
    # (e.g. wg cons buff = N*17408 bytes; 64KB L1 cap hit at N=4 with
    # margin but overflows hard at N>=5).
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

    KO_RMS = "llama_rmsnorm_int8.cc.o"
    KO_GEMM = "llama_gemm_int8_srs.cc.o"
    KO_ROPE = "llama_rope_int8.cc.o"
    KO_SILU = "llama_silu_mul_int8.cc.o"
    KO_FKV = "llama_flowkv.cc.o"
    KO_PT = "llama_layer_pt.cc.o"

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

    # --- Worker bodies (loop N_LAYERS times). ---
    # Router: 2-in (seed + back) -> 1-out (routed). Per-element copy
    # (D=64 ints) so the kernel-less router runs as scalar stores;
    # IRON's downstream multi-consumer broadcast handles fan-out to
    # rmsnorm1 + add1. First iter reads seed; rest read back (peeled
    # to avoid Python conditionals on a range_ loop var).
    def w_router(c_seed, c_back, p_routed):
        # L = 0
        x = c_seed.acquire(1)
        o = p_routed.acquire(1)
        for i in range(D):
            o[i] = x[i]
        p_routed.release(1)
        c_seed.release(1)
        # L = 1 .. N_LAYERS - 1
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

    # add2: writes to `back` for the first N_LAYERS-1 iterations
    # (residual loop), then to `out` on the final iteration.
    def w_add2_tail(c_a, c_b, c_back, c_out, k):
        for _ in range_(N_LAYERS - 1):
            a = c_a.acquire(1)
            b = c_b.acquire(1)
            o = c_back.acquire(1)
            k(a, b, o)
            c_a.release(1)
            c_b.release(1)
            c_back.release(1)
        # final layer
        a = c_a.acquire(1)
        b = c_b.acquire(1)
        o = c_out.acquire(1)
        k(a, b, o)
        c_a.release(1)
        c_b.release(1)
        c_out.release(1)

    # --- Workers (pinned to DECODE_PLACEMENT-ish tiles) ---
    workers = [
        # Router on spare tile (7, 4): 2-in (seed, back) -> 1-out (routed).
        Worker(
            w_router,
            [of_seed.cons(), of_back.cons(), of_routed.prod()],
            tile=Tile(7, 4),
        ),
        # rmsnorm1 (reads from routed; IRON broadcasts routed to add1 too)
        Worker(
            w_rms,
            [of_routed.cons(), of_gam_in.cons(), of_h1.prod(), k_rms],
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
            w_rope_fn,
            [of_qf.cons(), of_cs.cons(), of_qr.prod(), k_rope],
            tile=Tile(4, 4),
        ),
        # flowkv_qk
        Worker(
            w_qk_fn,
            [of_qr.cons(), of_kcache.cons(), of_probs.prod(), k_qk],
            tile=Tile(0, 4),
        ),
        # flowkv_sv
        Worker(
            w_sv_fn,
            [of_vcache.cons(), of_probs.cons(), of_af.prod(), k_sv],
            tile=Tile(0, 5),
        ),
        # o_proj
        Worker(
            w_gemm,
            [of_af.cons(), of_wo.cons(), of_op.prod(), k_gemm_DD],
            tile=Tile(3, 2),
        ),
        # add1 (broadcasts from routed -- same data as rmsnorm1's input)
        Worker(
            w_add_fn,
            [of_op.cons(), of_routed.cons(), of_x1.prod(), k_add],
            tile=Tile(6, 5),
        ),
        # rmsnorm2
        Worker(
            w_rms,
            [of_x1.cons(), of_gam_post.cons(), of_h2.prod(), k_rms],
            tile=Tile(6, 4),
        ),
        # gate_proj
        Worker(
            w_gemm,
            [of_h2.cons(), of_wg.cons(), of_gf.prod(), k_gemm_DHD],
            tile=Tile(4, 2),
        ),
        # up_proj
        Worker(
            w_gemm,
            [of_h2.cons(), of_wu.cons(), of_uf.prod(), k_gemm_DHD],
            tile=Tile(5, 2),
        ),
        # silu_mul
        Worker(
            w_silu_fn,
            [of_gf.cons(), of_uf.cons(), of_sf.prod(), k_silu],
            tile=Tile(4, 5),
        ),
        # down_proj
        Worker(
            w_gemm,
            [of_sf.cons(), of_wd.cons(), of_df.prod(), k_gemm_HDD],
            tile=Tile(6, 2),
        ),
        # add2: writes to back (loop) for layers 0..N-2, to out for layer N-1.
        Worker(
            w_add2_tail,
            [of_df.cons(), of_x1.cons(), of_back.prod(), of_out.prod(), k_add],
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

        seed_tg = rt.task_group()
        rt.fill(of_seed.prod(), xin, task_group=seed_tg)

        # SINGLE BD-chained TAP per fifo: outer dim walks the N per-layer
        # slices in one shim DMA task. Without this, N successive rt.fill
        # calls become N separate shim_dma start_tasks, which overflows
        # the AIE2P shim NoC per-channel task queue (depth 4) at N>=5 and
        # silently corrupts. With one BD-chained task per fifo, the shim
        # sees 1 task each (well under 4) regardless of N_LAYERS.
        #
        # AIE2P BD constraint: when multi-dim, each dim size must fit in
        # ~10 bits (<=1023). So per-layer nbytes > 1023 needs factoring
        # across two inner dims with stride=inner_size in the middle dim.
        def factor(nb):
            if nb <= 1023:
                return (1, nb)
            for inner in range(min(nb, 1023), 0, -1):
                if nb % inner == 0 and nb // inner <= 1023:
                    return (nb // inner, inner)
            raise ValueError(f"can't factor nbytes={nb} into two <=1023 dims")

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

        out_tg = rt.task_group()
        rt.drain(of_out.cons(), out, wait=True, task_group=out_tg)

        # Finish any remaining in-flight groups.
        rt.finish_task_group(seed_tg)
        for tg in tgs:
            rt.finish_task_group(tg)
        rt.finish_task_group(out_tg)

    return Program(NPU2(), rt).resolve_program()


def main():
    argparse.ArgumentParser().parse_args(sys.argv[1:])
    print(build())


if __name__ == "__main__":
    main()
