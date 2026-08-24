# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""One slot, three overlays, built with the `aie.iron.overlay` API.

This is the direct counterpart to `hw/one_slot.lit`'s hand-driven pipeline
(pm.py emit --recipe one_slot / pm.py link / pm.py emit again / pm.py check,
four separate RUN lines plus a hand-built `Config`/`Geometry`): here, the
device, the slot placement, the two-pass build, the per-overlay linking, and
the resident-stability check are all driven by one call,
`ProgramMemoryOverlayDesign(make_design).compile()`. `pmlib.workload` is
reused as-is for generating the three dummy, distinct-by-construction overlay
kernels -- that part is test fixture, not something the new API replaces.
"""

import argparse
import os

import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.device import AnyShimTile, Tile, from_name
from aie.iron.overlay import (
    ProgramMemoryOverlay,
    ProgramMemoryOverlayDesign,
    ProgramMemorySlot,
)
from aie.utils import set_current_device

from pmlib import workload

N_ELEMS = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True, help="comma-separated overlay tags")
    ap.add_argument("--out", required=True, help="work directory")
    args = ap.parse_args()
    tags = [int(t) for t in args.tags.split(",")]

    os.makedirs(args.out, exist_ok=True)
    object_files = []
    for tag in tags:
        out_o = os.path.join(args.out, f"w{tag}.o")
        workload.compile_overlay(
            tag, N_ELEMS, out_o, entry="overlay_entry_a", workdir=args.out
        )
        object_files.append(out_o)

    set_current_device(from_name("npu2", n_cols=2))

    tile_ty = np.ndarray[(N_ELEMS,), np.dtype[np.int32]]
    host_in_ty = np.ndarray[(N_ELEMS,), np.dtype[np.int32]]
    host_out_shape = (len(tags), N_ELEMS)
    host_out_ty = np.ndarray[host_out_shape, np.dtype[np.int32]]

    def make_design():
        compute_tile = Tile(0, 2)
        slot = ProgramMemorySlot("a", [tile_ty, tile_ty], tile=compute_tile, size=0x2000)
        overlays = [
            ProgramMemoryOverlay(f"w{tag}", slot, object_file_name=obj)
            for tag, obj in zip(tags, object_files)
        ]

        of_in = ObjectFifo(tile_ty, name="in")
        of_out = ObjectFifo(tile_ty, name="out")

        def core_fn(in_cons, out_prod, slot):
            for _ in range(len(tags)):
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
        )

        def sequence(host_in, host_out, in_prod, out_cons):
            for phase, overlay in enumerate(overlays):
                slot.load(overlay)
                tg = TaskGroup()
                in_prod.fill(host_in, group=tg)
                out_cons.drain(
                    host_out,
                    TensorAccessPattern(
                        host_out_shape, phase * N_ELEMS, [1, 1, 1, N_ELEMS], [0, 0, 0, 1]
                    ),
                    wait=True,
                    group=tg,
                )
                tg.finish()

        rt = Runtime(
            sequence,
            [
                host_in_ty,
                host_out_ty,
                of_in.prod(tile=AnyShimTile),
                of_out.cons(tile=AnyShimTile),
            ],
        )
        program = Program(from_name("npu2", n_cols=2), rt, workers=[worker])
        return program, overlays

    xclbin, insts = ProgramMemoryOverlayDesign(make_design).compile(
        work_dir=args.out,
        xclbin_path=os.path.join(args.out, "aie.xclbin"),
        insts_path=os.path.join(args.out, "insts.bin"),
    )
    print(f"built {xclbin} {insts}")


if __name__ == "__main__":
    main()
