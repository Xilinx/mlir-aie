"""Phase 6c.3b.1: FFN-half at production Llama 3.2 1B shapes (D=2048, HD=8192).

Six-worker xclbin wiring:
  rmsnorm2 -> [gate_proj | up_proj] -> silu_mul -> down_proj -> add2

All projections use the tiled-GEMV kernel (N_TILE=4 rows per tile call,
weights stream chunk-by-chunk from DRAM via BD-chained TAPs). Same
runtime + L1 budget as the standalone tiled GEMM test, just three of them
chained in one design. Input `x1_in` is consumed by rmsnorm2 AND fed to
add2 as residual (multi-consumer ObjectFifo).
"""

import argparse
import sys

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern
from aie.dialects.arith import index_cast
from aie.ir import IntegerType


def _i32(idx):
    """Cast an index-typed SSA value to i32 (needed for kernels declared with np.int32)."""
    return index_cast(IntegerType.get_signless(32), idx)


# --- Production Llama 3.2 1B FFN shapes ---
D       = 2048
HD      = 8192
N_TILE  = 4

# Per-tile slot sizes:
#   gate: [N_TILE*K i8 weights | N_TILE i32 bias | N_TILE fp32 w_scales]
#           — uses existing _perchan kernel (scales as closure args). Gate's
#           output scale is locked to the silu LUT's baked gate_scale and
#           thus doesn't need to be dynamic.
#   up:   [16 B prefix | N_TILE*K weights | N_TILE i32 bias | N_TILE fp32 w_scales]
#           — uses _perchan_v2_up kernel. Prefix = (up_act_scale fp32,
#           up_inv_out_scale fp32, silu_up_scale fp32, silu_inv_out_scale fp32).
#           Kernel also mirrors the silu 8 B into of_uf's tail.
#   down: [8 B prefix | N_TILE*K weights | N_TILE i32 bias | N_TILE fp32 w_scales]
#           — uses _perchan_v2 kernel. Prefix = (down_act_scale fp32,
#           down_inv_out_scale fp32).
# Slot prefixes must be multiples of 64 B so the weight body (which the
# kernel hits with aie::load_v<64>) stays 64-byte-aligned. Smaller prefixes
# silently mis-load weights — the L1-resident buffer is 64-aligned at the
# slot base, so `w_tile + kPrefix` is only aligned when kPrefix % 64 == 0.
#
# Gate: no prefix (closure scales, body starts at slot[0]).
# Up:   64 B prefix = 8 B (up's own scales) + 8 B (silu scales) + 48 B pad.
# Down: 64 B prefix = 8 B (down's scales) + 56 B pad.
PREFIX_ALIGN = 64
WU_PREFIX = PREFIX_ALIGN
WD_PREFIX = PREFIX_ALIGN
WG_SLOT = N_TILE * D + N_TILE * 4 + N_TILE * 4         # gate (K=D, N=HD)
WU_SLOT = WU_PREFIX + N_TILE * D  + N_TILE * 4 + N_TILE * 4
WD_SLOT = WD_PREFIX + N_TILE * HD + N_TILE * 4 + N_TILE * 4

# Up's output fifo carries silu's 8 B scales after the HD-byte gemm output.
UF_BYTES = HD + 8

N_TILES_GATE = HD // N_TILE             # 2048 tile iterations
N_TILES_UP   = HD // N_TILE
N_TILES_DOWN = D  // N_TILE             # 512 tile iterations

WG_TOTAL = N_TILES_GATE * WG_SLOT
WU_TOTAL = N_TILES_UP   * WU_SLOT
WD_TOTAL = N_TILES_DOWN * WD_SLOT

GAMMA_BYTES = D * 2                     # bf16

# Weights blob layout.
OFF_GAMMA = 0
OFF_WG    = OFF_GAMMA + GAMMA_BYTES
OFF_WU    = OFF_WG    + WG_TOTAL
OFF_WD    = OFF_WU    + WU_TOTAL
WEIGHTS_BYTES = OFF_WD + WD_TOTAL

# Baked-at-build-time scales (Phase 6c.5b.2):
# rmsnorm and gate-gemm both need scales fixed at MLIR-emit time. The
# silu LUT is precomputed via gen_silu_lut.py using SILU_GATE_SCALE; gate's
# inv_out_scale is therefore locked to 1/SILU_GATE_SCALE so the LUT keys
# line up with gate's int8 output.
#
# Up's act_scale and inv_out_scale, silu's two scales, and down's pair are
# all delivered at dispatch time via slot prefixes (see WU_SLOT/WD_SLOT).
import os as _os
def _envf(name, default):
    return float(_os.environ.get(name, default))

ACT_SCALE         = _envf("FFN_ACT_SCALE",     "0.05")
INV_ACT_SCALE     = _envf("FFN_INV_ACT_SCALE", str(1.0 / 0.05))

# Matches the build's SILU_GATE_SCALE (Makefile default 0.05). Gate's
# output is requantized at this scale every dispatch so the silu LUT
# stays valid without rebuild.
SILU_GATE_SCALE    = _envf("SILU_GATE_SCALE", "0.05")
GATE_INV_OUT_SCALE = 1.0 / SILU_GATE_SCALE

# Legacy constants — kept so aie2_ffn_half_gate_only.py (debug) still
# imports cleanly. The main build() no longer reads them.
UP_INV_OUT_SCALE   = _envf("FFN_UP_INV_OUT_SCALE",   "1.0")
SILU_IN_SCALE      = _envf("FFN_SILU_IN_SCALE",      str(SILU_GATE_SCALE))
SILU_INV_OUT_SCALE = _envf("FFN_SILU_INV_OUT_SCALE", str(1.0 / SILU_GATE_SCALE))
DOWN_ACT_SCALE     = _envf("FFN_DOWN_ACT_SCALE",     "0.05")
DOWN_INV_OUT_SCALE = _envf("FFN_DOWN_INV_OUT_SCALE", "1.0")


def _i8(n):
    return np.ndarray[(int(n),), np.dtype[np.int8]]


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
    # Debug: set FFN_DEBUG_DRAIN=gf|uf|sf|df adds a SECOND runtime out
    # buffer (D bytes) that carries the first D bytes of the chosen
    # intermediate fifo. The chain still runs end-to-end (silu/down/add
    # all execute); the debug drain is broadcast-fanned off the chosen
    # fifo so the downstream consumer still sees the same bytes.
    debug_drain = _os.environ.get("FFN_DEBUG_DRAIN", "")
    rt_xin_ty   = _i8(D)
    rt_w_ty     = _i8(WEIGHTS_BYTES)
    rt_out_ty   = _i8(D)
    rt_debug_ty = _i8(D)

    t_D_i8     = _i8(D)
    t_HD_i8    = _i8(HD)
    t_UF_i8    = _i8(UF_BYTES)        # HD + 8 (silu scales tail)
    t_D_bf16   = _bf16(D)
    t_WG_slot  = _i8(WG_SLOT)
    t_WU_slot  = _i8(WU_SLOT)
    t_WD_slot  = _i8(WD_SLOT)

    # --- ObjectFifos ---
    # Residual input -> rmsnorm AND add2 (multi-consumer broadcast).
    # Multi-consumer broadcasts must be depth=1 (Bug 6: depth=2 broadcast
    # desyncs between back-to-back dispatches, producing alternating-iter
    # garbage on same-xclbin-multi-dispatch).
    of_xin = ObjectFifo(t_D_i8, depth=1, name="xin")
    of_h2     = ObjectFifo(t_D_i8,    depth=1, name="h2")   # rmsnorm out -> gate AND up
    of_gamma  = ObjectFifo(t_D_bf16,  depth=1, name="gamma")
    of_wg     = ObjectFifo(t_WG_slot, depth=2, name="wg")
    of_wu     = ObjectFifo(t_WU_slot, depth=2, name="wu")
    of_wd     = ObjectFifo(t_WD_slot, depth=1, name="wd")   # K=8192 needs depth=1
    of_gf     = ObjectFifo(t_HD_i8,   depth=1, name="gf")   # full HD-vector after gate
    of_uf     = ObjectFifo(t_UF_i8,   depth=1, name="uf")   # HD + 8 B silu scales tail
    of_sf     = ObjectFifo(t_HD_i8,   depth=1, name="sf")
    of_df     = ObjectFifo(t_D_i8,    depth=1, name="df")
    of_out    = ObjectFifo(t_D_i8,    depth=1, name="layer_out")

    # gate/up/down each have their own per-tile output fifo that the
    # shim drain joins back into the full HD- or D-sized buffer the
    # downstream consumer's worker reads. We use intermediate "tile-out"
    # fifos and let the downstream worker reassemble by acquiring once
    # per tile. Equivalent: bypass tile-fifos by making the gemm worker
    # produce DIRECTLY into the HD-sized of_gf/uf/df fifo, writing the
    # right slice each tile iter. IRON allows acquiring the full HD
    # buffer once at start and indexing it in the per-tile loop.
    # That's cleaner; do it that way.

    KO_RMS  = "llama_rmsnorm_int8.cc.o"
    KO_GEMM = "llama_gemm_int8_srs_tiled_ffn.cc.o"
    KO_SILU = "llama_silu_mul_int8.cc.o"
    KO_PT   = "llama_layer_pt.cc.o"

    k_rms      = Kernel("llama_rmsnorm_int8", KO_RMS,
                        [t_D_i8, t_D_bf16, t_D_i8, np.float32, np.float32])
    # Gate: existing _perchan kernel (scales baked as closure args). Gate's
    # output scale = 1/SILU_GATE_SCALE so it matches the silu LUT.
    k_gemm_gate = Kernel("llama_gemm_tiled_K2048_N4_perchan", KO_GEMM,
                         [t_D_i8,  t_WG_slot, t_HD_i8, np.int32,
                          np.float32, np.float32])
    # Up: _perchan_v2_up — reads gemm scales + silu scales from 16 B slot
    # prefix; writes silu scales to of_uf's 8 B tail. 4 args (no scalars).
    k_gemm_up   = Kernel("llama_gemm_tiled_K2048_N4_perchan_v2_up", KO_GEMM,
                         [t_D_i8,  t_WU_slot, t_UF_i8, np.int32])
    # Down: _perchan_v2 — reads gemm scales from 8 B slot prefix.
    k_gemm_down = Kernel("llama_gemm_tiled_K8192_N4_perchan_v2", KO_GEMM,
                         [t_HD_i8, t_WD_slot, t_D_i8,  np.int32])
    # Silu: _dyn — reads (up_scale, inv_out_scale) from up[HD..HD+8].
    k_silu      = Kernel("llama_silu_mul_int8_dyn", KO_SILU,
                         [t_HD_i8, t_UF_i8, t_HD_i8])
    k_add       = Kernel("llama_pt_add_D", KO_PT, [t_D_i8, t_D_i8, t_D_i8])

    # --- Worker bodies ---
    def w_rms(c_in, c_gamma, c_out, k):
        x = c_in.acquire(1); g = c_gamma.acquire(1); o = c_out.acquire(1)
        k(x, g, o, ACT_SCALE, INV_ACT_SCALE)
        c_in.release(1); c_gamma.release(1); c_out.release(1)

    # Gate gemm worker. Closure-baked scales (ACT_SCALE, GATE_INV_OUT_SCALE).
    def w_gemm_gate(c_act, c_w, c_out, k):
        a = c_act.acquire(1)
        o = c_out.acquire(1)
        for t in range_(N_TILES_GATE):
            w = c_w.acquire(1)
            k(a, w, o, _i32(t), ACT_SCALE, GATE_INV_OUT_SCALE)
            c_w.release(1)
        c_out.release(1); c_act.release(1)

    # Up gemm worker (v2_up). No scalar args — kernel reads everything
    # from the 16 B slot prefix.
    def w_gemm_up(c_act, c_w, c_out, k):
        a = c_act.acquire(1)
        o = c_out.acquire(1)
        for t in range_(N_TILES_UP):
            w = c_w.acquire(1)
            k(a, w, o, _i32(t))
            c_w.release(1)
        c_out.release(1); c_act.release(1)

    # Down gemm worker (v2). No scalar args — kernel reads 32 B slot prefix.
    def w_gemm_down(c_act, c_w, c_out, k):
        a = c_act.acquire(1)
        o = c_out.acquire(1)
        for t in range_(N_TILES_DOWN):
            w = c_w.acquire(1)
            k(a, w, o, _i32(t))
            c_w.release(1)
        c_out.release(1); c_act.release(1)

    def w_silu(c_g, c_u, c_out, k):
        g = c_g.acquire(1); u = c_u.acquire(1); o = c_out.acquire(1)
        k(g, u, o)
        c_g.release(1); c_u.release(1); c_out.release(1)

    def w_add(c_a, c_b, c_out, k):
        a = c_a.acquire(1); b = c_b.acquire(1); o = c_out.acquire(1)
        k(a, b, o)
        c_a.release(1); c_b.release(1); c_out.release(1)

    # Per-channel kernel uses ~2-3 KB stack (int32 acc unpack + fp32 chain
    # + spills). The 1 KB default overflows silently when chained -- Peano
    # Bug 5 again. Bump every perchan worker explicitly.
    PERCHAN_STACK = 8192
    workers = [
        Worker(w_rms, [of_xin.cons(), of_gamma.cons(), of_h2.prod(), k_rms],
               tile=Tile(5, 4)),
        Worker(w_gemm_gate,
               [of_h2.cons(), of_wg.cons(), of_gf.prod(), k_gemm_gate],
               tile=Tile(0, 2), stack_size=PERCHAN_STACK),
        Worker(w_gemm_up,
               [of_h2.cons(), of_wu.cons(), of_uf.prod(), k_gemm_up],
               tile=Tile(1, 2), stack_size=PERCHAN_STACK),
        Worker(w_silu, [of_gf.cons(), of_uf.cons(), of_sf.prod(), k_silu],
               tile=Tile(2, 4), stack_size=8192),
        Worker(w_gemm_down,
               [of_sf.cons(), of_wd.cons(), of_df.prod(), k_gemm_down],
               tile=Tile(6, 2), stack_size=PERCHAN_STACK),
        Worker(w_add, [of_df.cons(), of_xin.cons(), of_out.prod(), k_add],
               tile=Tile(7, 5)),
    ]

    rt = Runtime()
    seq_tys = (rt_xin_ty, rt_w_ty, rt_out_ty)
    if debug_drain:
        seq_tys = (rt_xin_ty, rt_w_ty, rt_out_ty, rt_debug_ty)
    with rt.sequence(*seq_tys) as seq_args:
        xin = seq_args[0]; wblob = seq_args[1]; out = seq_args[2]
        debug_out = seq_args[3] if debug_drain else None
        rt.start(*workers)

        # x1_in: single fill, broadcasts to rmsnorm AND add2.
        xin_tg = rt.task_group()
        rt.fill(of_xin.prod(), xin, task_group=xin_tg)

        # gamma_post: pull from wblob[0:GAMMA_BYTES].
        gam_tg = rt.task_group()
        rt.fill(of_gamma.prod(), wblob,
                tap=strided_tap(WEIGHTS_BYTES, OFF_GAMMA, GAMMA_BYTES,
                                GAMMA_BYTES, 1),
                task_group=gam_tg)

        # gate, up, down: BD-chained TAP per matrix.
        wg_tg = rt.task_group()
        rt.fill(of_wg.prod(), wblob,
                tap=strided_tap(WEIGHTS_BYTES, OFF_WG, WG_SLOT,
                                WG_SLOT, N_TILES_GATE),
                task_group=wg_tg)

        wu_tg = rt.task_group()
        rt.fill(of_wu.prod(), wblob,
                tap=strided_tap(WEIGHTS_BYTES, OFF_WU, WU_SLOT,
                                WU_SLOT, N_TILES_UP),
                task_group=wu_tg)

        wd_tg = rt.task_group()
        rt.fill(of_wd.prod(), wblob,
                tap=strided_tap(WEIGHTS_BYTES, OFF_WD, WD_SLOT,
                                WD_SLOT, N_TILES_DOWN),
                task_group=wd_tg)

        out_tg = rt.task_group()
        rt.drain(of_out.cons(), out, wait=True, task_group=out_tg)
        if debug_drain:
            debug_tg = rt.task_group()
            target = {"gf": of_gf, "uf": of_uf, "sf": of_sf, "df": of_df}[debug_drain]
            tap = (strided_tap(HD, 0, D, D, 1) if debug_drain in ("gf","uf","sf")
                   else None)
            if tap is not None:
                rt.drain(target.cons(), debug_out, tap=tap,
                         wait=True, task_group=debug_tg)
            else:
                rt.drain(target.cons(), debug_out,
                         wait=True, task_group=debug_tg)

        rt.finish_task_group(xin_tg)
        rt.finish_task_group(gam_tg)
        rt.finish_task_group(wg_tg)
        rt.finish_task_group(wu_tg)
        rt.finish_task_group(wd_tg)
        rt.finish_task_group(out_tg)
        if debug_drain:
            rt.finish_task_group(debug_tg)

    return Program(NPU2(), rt).resolve_program()


def main():
    argparse.ArgumentParser().parse_args(sys.argv[1:])
    print(build())


if __name__ == "__main__":
    main()
