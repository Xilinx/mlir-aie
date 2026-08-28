# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""One slot, one overlay, built with the `aie.iron.overlay` API -- just enough
design to exercise `ProgramMemoryOverlayDesign.compile()`'s stack-budget
guard (`design.py`'s `_check_stack_budget`) against a real aiecc-built
resident and a real linked overlay.

Counterpart to `build/stack_budget.lit`, which drives the same rule (resident
frame + overlay frame must fit the stack budget) through `pm.py stack`
against hand-supplied objects instead.
"""

import argparse

import numpy as np
from aie.iron import ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.device import AnyShimTile, Tile, from_name
from aie.iron.overlay import (
    ProgramMemoryOverlay,
    ProgramMemoryOverlayDesign,
    ProgramMemorySlot,
)
from aie.utils import set_current_device

N_ELEMS = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay-obj", required=True)
    ap.add_argument("--stack-size", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_current_device(from_name("npu2", n_cols=2))
    tile_ty = np.ndarray[(N_ELEMS,), np.dtype[np.int32]]
    host_out_ty = np.ndarray[(1, N_ELEMS), np.dtype[np.int32]]

    def make_design():
        compute_tile = Tile(0, 2)
        slot = ProgramMemorySlot("a", [tile_ty, tile_ty], tile=compute_tile, size=0x2000)
        overlay = ProgramMemoryOverlay("ovl", slot, object_file_name=args.overlay_obj)

        of_in = ObjectFifo(tile_ty, name="in")
        of_out = ObjectFifo(tile_ty, name="out")

        def core_fn(in_cons, out_prod, slot):
            slot.wait()
            a = in_cons.acquire(1)
            c = out_prod.acquire(1)
            slot(a, c)
            in_cons.release(1)
            out_prod.release(1)

        worker = Worker(
            core_fn,
            [of_in.cons(), of_out.prod(), slot],
            tile=compute_tile,
            while_true=False,
            stack_size=args.stack_size,
        )

        def sequence(host_in, host_out, in_prod, out_cons):
            slot.load(overlay)
            tg = TaskGroup()
            in_prod.fill(host_in, group=tg)
            out_cons.drain(host_out, wait=True, group=tg)
            tg.finish()

        rt = Runtime(
            sequence,
            [
                tile_ty,
                host_out_ty,
                of_in.prod(tile=AnyShimTile),
                of_out.cons(tile=AnyShimTile),
            ],
        )
        program = Program(from_name("npu2", n_cols=2), rt, workers=[worker])
        return program, [overlay]

    xclbin, insts = ProgramMemoryOverlayDesign(make_design).compile(work_dir=args.out)
    print(f"built {xclbin} {insts}")


if __name__ == "__main__":
    main()
