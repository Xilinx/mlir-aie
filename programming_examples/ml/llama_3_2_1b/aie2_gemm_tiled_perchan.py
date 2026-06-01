"""Phase 6c.5b.1a: per-channel-weight tiled GEMM design (standalone).

Same dataflow as aie2_gemm_tiled.py (1 CT, act resident, weights stream
in per-tile slots), but the slot carries per-output-channel fp32 scales
and the kernel does fp32 dequant + re-quant instead of a single banker_srs.

Per-tile slot layout:
    [N_TILE * K i8 weights | N_TILE i32 bias | N_TILE fp32 w_scales]

Kernel takes (act_scale, inv_out_scale) as fp32 closure constants baked
at IRON-emit time — for this standalone test the values come from
calibration done by the test BEFORE invoking the design (so the kernel
build matches the data the test generates). Dynamic per-call scales is
6c.5b.2 work.
"""

import argparse
import os
import sys

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern
from aie.dialects.arith import constant, index_cast
from aie.ir import IntegerType


def _i32_const(v):
    return constant(IntegerType.get_signless(32), int(v))


def _i32(idx):
    return index_cast(IntegerType.get_signless(32), idx)


GEMM_COL, GEMM_ROW = 0, 2


def build(K: int, N: int, n_tile: int,
          act_scale: float, inv_out_scale: float):
    assert N % n_tile == 0, f"N={N} must be a multiple of N_TILE={n_tile}"
    n_tiles = N // n_tile
    slot_bytes = n_tile * K + n_tile * 4 + n_tile * 4   # weights + bias + w_scales

    act_ty     = np.ndarray[(K,),                np.dtype[np.int8]]
    w_ty       = np.ndarray[(slot_bytes,),       np.dtype[np.int8]]
    out_ty     = np.ndarray[(N,),                np.dtype[np.int8]]  # full N, NOT per-tile
    w_blob_ty  = np.ndarray[(n_tiles * slot_bytes,), np.dtype[np.int8]]
    out_buf_ty = np.ndarray[(N,),                np.dtype[np.int8]]

    of_act = ObjectFifo(act_ty, depth=1, name="act")
    # Output fifo carries the FULL N-byte vector (not per-tile) because
    # the per-channel kernel uses the (out_full, tile_idx) slicing
    # pattern from FFN-half: it writes into out_full[tile_idx*N_TILE..]
    # each call. Acquired once outside the loop and drained as one
    # buffer to the host.
    of_out = ObjectFifo(out_ty, depth=1, name="out")

    # Match the L1 budget rules from aie2_gemm_tiled.py — at K=8192 slot
    # exceeds one bank, force depth=1 (our local mlir-aie patch's
    # multi-bank spanning handles the layout).
    BANK = 16 * 1024
    used2 = K + 2 * slot_bytes + 2 * n_tile + 1024
    w_depth = 2 if (slot_bytes <= BANK and used2 <= 48 * 1024) else 1
    of_w = ObjectFifo(w_ty, depth=w_depth, name="w")

    # Kernel symbol: gate/up shape uses K2048, down uses K8192. Standalone
    # test exercises whichever K the caller picks.
    if K == 2048:
        sym = "llama_gemm_tiled_K2048_N4_perchan"
    elif K == 8192:
        sym = "llama_gemm_tiled_K8192_N4_perchan"
    else:
        raise ValueError(f"K={K} not supported by per-channel kernel")

    kernel = Kernel(
        sym, "llama_gemm_int8_srs_tiled_ffn.cc.o",
        [act_ty, w_ty, out_ty, np.int32, np.float32, np.float32],
    )

    def core_fn(c_act, c_w, c_out, k):
        a = c_act.acquire(1)
        o = c_out.acquire(1)              # acquire ONCE -- full N bytes
        for t in range_(n_tiles):
            w = c_w.acquire(1)
            k(a, w, o, _i32(t), act_scale, inv_out_scale)
            c_w.release(1)
        c_out.release(1); c_act.release(1)

    worker = Worker(
        core_fn,
        [of_act.cons(), of_w.cons(), of_out.prod(), kernel],
        tile=Tile(GEMM_COL, GEMM_ROW),
        stack_size=4096,
    )

    def factor(nb):
        if nb <= 1023: return (1, nb)
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

    rt = Runtime()
    with rt.sequence(act_ty, w_blob_ty, out_buf_ty) as (a, w_blob, o_buf):
        rt.start(worker)
        act_tg = rt.task_group()
        rt.fill(of_act.prod(), a, task_group=act_tg)
        w_tg = rt.task_group()
        rt.fill(of_w.prod(), w_blob,
                tap=strided_tap(n_tiles * slot_bytes, 0, slot_bytes,
                                slot_bytes, n_tiles),
                task_group=w_tg)
        # Drain the full N-byte output in one shim DMA (no per-tile
        # strided pattern needed; kernel writes all tiles into one
        # of_out slot).
        o_tg = rt.task_group()
        rt.drain(of_out.cons(), o_buf,
                 tap=strided_tap(N, 0, N, N, 1),
                 wait=True, task_group=o_tg)
        rt.finish_task_group(act_tg)
        rt.finish_task_group(w_tg)
        rt.finish_task_group(o_tg)

    return Program(NPU2(), rt).resolve_program()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-K", type=int, default=2048)
    p.add_argument("-N", type=int, default=64)
    p.add_argument("--n-tile", type=int, default=4)
    p.add_argument("--act-scale", type=float,
                   default=float(os.environ.get("LLAMA_GEMM_ACT_SCALE", "0.05")))
    p.add_argument("--inv-out-scale", type=float,
                   default=float(os.environ.get("LLAMA_GEMM_INV_OUT_SCALE", "1.0")))
    args = p.parse_args(sys.argv[1:])
    print(build(args.K, args.N, args.n_tile,
                args.act_scale, args.inv_out_scale))


if __name__ == "__main__":
    main()
