# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Ping-pong, built with the `aie.iron.overlay` API.

Direct counterpart to `hw/pingpong.lit`'s hand-driven pipeline (pm.py emit
--recipe pingpong / pm.py link x3 (both overlays + the stub) / pm.py emit
again / pm.py order / pm.py check --overlays ...): here,
`ProgramMemorySlot.pingpong()` computes the geometry, builds and links the
bootstrap park, and `ProgramMemoryOverlayDesign` drives both passes -- no
Geometry, no slot.ld, no stub.cc wired in by hand, no manual npu.maskpoll.
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
SLOT_SIZE = 0x1C00


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="work directory")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    wb_o = os.path.join(args.out, "wb.o")
    wa_o = os.path.join(args.out, "wa.o")
    workload.compile_overlay(21, N_ELEMS, wb_o, entry="overlay_entry_b", workdir=args.out)
    workload.compile_overlay(42, N_ELEMS, wa_o, entry="overlay_entry_a", workdir=args.out)

    set_current_device(from_name("npu2", n_cols=2))

    tile_ty = np.ndarray[(N_ELEMS,), np.dtype[np.int32]]
    host_in_ty = np.ndarray[(N_ELEMS,), np.dtype[np.int32]]
    n_phases = 8
    host_out_shape = (n_phases, N_ELEMS)
    host_out_ty = np.ndarray[host_out_shape, np.dtype[np.int32]]

    def make_design():
        compute_tile = Tile(0, 2)
        slot_a, slot_b = ProgramMemorySlot.pingpong(
            "a", "b", [tile_ty, tile_ty], tile=compute_tile, size=SLOT_SIZE
        )
        k_b = ProgramMemoryOverlay("wb", slot_b, object_file_name=wb_o)
        k_a = ProgramMemoryOverlay("wa", slot_a, object_file_name=wa_o)
        # Phase 0 must land on the high-granule slot (slot_b): the core boots
        # straight into the resident, so its first wait() must be the normal
        # in-resident one, not a jump into the not-yet-written bootstrap park.
        schedule = [k_b, k_a] * (n_phases // 2)

        of_in = ObjectFifo(tile_ty, name="in")
        of_out = ObjectFifo(tile_ty, name="out")

        def core_fn(in_cons, out_prod, slot_a, slot_b):
            slots = {id(k_b): slot_b, id(k_a): slot_a}
            for ovl in schedule:
                s = slots[id(ovl)]
                s.wait()
                a = in_cons.acquire(1)
                c = out_prod.acquire(1)
                s(a, c)
                in_cons.release(1)
                out_prod.release(1)

        worker = Worker(
            core_fn,
            [of_in.cons(), of_out.prod(), slot_a, slot_b],
            tile=compute_tile,
            while_true=False,
        )

        def sequence(host_in, host_out, in_prod, out_cons):
            for phase, ovl in enumerate(schedule):
                ovl.slot.load(ovl)
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
        return program, [k_b, k_a]

    xclbin, insts = ProgramMemoryOverlayDesign(make_design).compile(
        work_dir=args.out,
        xclbin_path=os.path.join(args.out, "aie.xclbin"),
        insts_path=os.path.join(args.out, "insts.bin"),
    )
    print(f"built {xclbin} {insts}")


if __name__ == "__main__":
    main()
