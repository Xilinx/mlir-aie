"""Phase 6c.1 tiled decode GEMM design.

1 CT. Activation acquired ONCE (kept resident in L1), weights stream
through a depth=2 ObjectFifo with N/N_TILE successive slots, output
streams out the same way. Single BD-chained shim DMA for weights and
for the output drain (collapsed into one task each, well under the
shim NoC per-channel queue depth).

The host packs weights+bias as N/N_TILE successive [N_TILE*K i8 |
N_TILE i32 bias] slots. The kernel's signature matches the legacy
gemm: act / w_tile / out_tile / right_shift.
"""

import argparse
import sys

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern

GEMM_COL, GEMM_ROW = 0, 2

RIGHT_SHIFT = 12


def build(K: int, N: int, n_tile: int, w_depth: int = None, split_bias: bool = None):
    assert N % n_tile == 0, f"N={N} must be a multiple of N_TILE={n_tile}"
    n_tiles = N // n_tile

    # Always use inlined bias: separating bias to its own fifo would
    # push the CT to 3 input DMA channels (act + w + b), exceeding the
    # 2-in budget. Instead, we tune `w_depth` and (if needed) drop to
    # depth=1 when the inlined slot doesn't have a bank-friendly size.
    split_bias = False
    BANK = 16 * 1024

    weight_slot_bytes = n_tile * K
    bias_slot_bytes = n_tile * 4
    inlined_slot_bytes = weight_slot_bytes + bias_slot_bytes

    if w_depth is None:
        # depth=2 only works if both slots are bank-friendly. The bug
        # signature: inlined slot is >1 bank but not a clean multiple
        # of a bank (e.g. 32784 = 2 banks + 16 B). Bank-aware fails,
        # sequential fallback overlaps with act -> Peano Bug 3a.
        bank_ok = (inlined_slot_bytes <= BANK) or (inlined_slot_bytes % BANK == 0)
        # Even when bank_ok, total L1 has to fit.
        used2 = K + 2 * inlined_slot_bytes + 2 * n_tile + 1024
        w_depth = 2 if (bank_ok and used2 <= 48 * 1024) else 1

    act_ty = np.ndarray[(K,), np.dtype[np.int8]]
    out_ty = np.ndarray[(n_tile,), np.dtype[np.int8]]
    out_buf_ty = np.ndarray[(N,), np.dtype[np.int8]]

    of_act = ObjectFifo(act_ty, depth=1, name="act")
    of_out = ObjectFifo(out_ty, depth=2, name="out")

    w_ty = np.ndarray[(inlined_slot_bytes,), np.dtype[np.int8]]
    w_blob_ty = np.ndarray[(n_tiles * inlined_slot_bytes,), np.dtype[np.int8]]

    of_w = ObjectFifo(w_ty, depth=w_depth, name="w")

    kernel = Kernel(
        "llama_gemm_int8_srs_tiled",
        "llama_gemm_int8_srs_tiled.cc.o",
        [act_ty, w_ty, out_ty, np.int32],
    )

    def core_fn(c_act, c_w, c_out, k):
        a = c_act.acquire(1)
        for _ in range_(n_tiles):
            w = c_w.acquire(1)
            o = c_out.acquire(1)
            k(a, w, o, RIGHT_SHIFT)
            c_w.release(1)
            c_out.release(1)
        c_act.release(1)

    worker = Worker(
        core_fn,
        [of_act.cons(), of_w.cons(), of_out.prod(), kernel],
        tile=Tile(GEMM_COL, GEMM_ROW),
    )

    # AIE2P BD constraint: each dim size <= 1023 when multi-dim. Factor
    # the per-slot transfer into two inner dims. Same helper used by
    # aie2_chain_real.py.
    def factor(nb):
        if nb <= 1023:
            return (1, nb)
        for inner in range(min(nb, 1023), 0, -1):
            if nb % inner == 0 and nb // inner <= 1023:
                return (nb // inner, inner)
        raise ValueError(f"can't factor nbytes={nb} into two <=1023 dims")

    def strided_tap(total_bytes, base_off, per_slot_stride, slot_bytes, n_slots):
        outer, inner = factor(slot_bytes)
        return TensorAccessPattern(
            tensor_dims=[total_bytes],
            offset=base_off,
            sizes=[1, n_slots, outer, inner],
            strides=[0, per_slot_stride, inner, 1],
        )

    blob_total = n_tiles * inlined_slot_bytes

    rt = Runtime()
    with rt.sequence(act_ty, w_blob_ty, out_buf_ty) as (a, w_blob, o_buf):
        rt.start(worker)
        act_tg = rt.task_group()
        rt.fill(of_act.prod(), a, task_group=act_tg)

        w_tg = rt.task_group()
        rt.fill(
            of_w.prod(),
            w_blob,
            tap=strided_tap(
                blob_total, 0, inlined_slot_bytes, inlined_slot_bytes, n_tiles
            ),
            task_group=w_tg,
        )

        o_tg = rt.task_group()
        rt.drain(
            of_out.cons(),
            o_buf,
            tap=strided_tap(N, 0, n_tile, n_tile, n_tiles),
            wait=True,
            task_group=o_tg,
        )

        rt.finish_task_group(act_tg)
        rt.finish_task_group(w_tg)
        rt.finish_task_group(o_tg)

    return Program(NPU2(), rt).resolve_program()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-K", type=int, default=2048)
    p.add_argument("-N", type=int, default=64)
    p.add_argument("--n-tile", type=int, default=8)
    args = p.parse_args(sys.argv[1:])
    print(build(args.K, args.N, args.n_tile))


if __name__ == "__main__":
    main()
