# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""One slot written by a neighboring compute tile's own DMA, not the host.

Direct counterpart to test/npu-xrt/tile_sourced_ctrl_pkt_spike/aie.mlir, at
real ProgramMemorySlot scale instead of a 2-word proof: a source Worker's
core_fn calls `slot.load(overlay)` once, and the target Worker's `wait()`
polls the plain flag Buffer the source's control-packet burst sets on
completion (there is no runtime sequence in this transport to release a
host-driven barrier).

Slot size is small (0x90 = 144 bytes, 9 control-packet chunks), not
hw/one_slot.lit's 0x2000: a source tile's entire BD table is 16 descriptors
(`AIETargetModel::getNumBDs()`), one per control-packet chunk (see
ProgramMemorySlot._load_tile_sourced's comments -- BdIteration, the obvious
way to cover more chunks with fewer descriptors, was hardware-verified to
corrupt the packet's embedded address on every execution but the first once
combined with packet-tagging). ~144 bytes is this transport's real, current
ceiling, not a toy size chosen for convenience -- a documented v1 scope, the
same spirit as this transport's one-`load()`-call limit.
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

N_ELEMS = 4
SLOT_SIZE = 0x90


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=int, required=True, help="overlay tag")
    ap.add_argument("--out", required=True, help="work directory")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    out_o = os.path.join(args.out, f"w{args.tag}.o")
    workload.compile_overlay_of_size(
        args.tag,
        N_ELEMS,
        out_o,
        SLOT_SIZE,
        entry="overlay_entry_a",
        workdir=args.out,
    )

    set_current_device(from_name("npu2", n_cols=2))

    tile_ty = np.ndarray[(N_ELEMS,), np.dtype[np.int32]]
    host_in_ty = np.ndarray[(N_ELEMS,), np.dtype[np.int32]]
    host_out_ty = np.ndarray[(N_ELEMS,), np.dtype[np.int32]]

    def make_design():
        target_tile = Tile(0, 2)
        source_tile = Tile(1, 2)

        def source_core_fn():
            slot.load(overlay)

        source_worker = Worker(
            source_core_fn, [], tile=source_tile, while_true=False
        )

        slot = ProgramMemorySlot(
            "a",
            [tile_ty, tile_ty],
            tile=target_tile,
            size=SLOT_SIZE,
            source=source_worker,
        )
        overlay = ProgramMemoryOverlay(f"w{args.tag}", slot, object_file_name=out_o)

        of_in = ObjectFifo(tile_ty, name="in")
        of_out = ObjectFifo(tile_ty, name="out")

        def target_core_fn(in_cons, out_prod, slot):
            slot.wait()
            a = in_cons.acquire(1)
            c = out_prod.acquire(1)
            slot(a, c)
            in_cons.release(1)
            out_prod.release(1)

        target_worker = Worker(
            target_core_fn,
            [of_in.cons(), of_out.prod(), slot],
            tile=target_tile,
            while_true=False,
        )

        def sequence(host_in, host_out, in_prod, out_cons):
            tg = TaskGroup()
            in_prod.fill(host_in, group=tg)
            out_cons.drain(
                host_out,
                TensorAccessPattern((N_ELEMS,), 0, [1, 1, 1, N_ELEMS], [0, 0, 0, 1]),
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
        program = Program(
            from_name("npu2", n_cols=2), rt, workers=[target_worker, source_worker]
        )
        return program, [overlay]

    xclbin, insts = ProgramMemoryOverlayDesign(make_design).compile(
        work_dir=args.out,
        xclbin_path=os.path.join(args.out, "aie.xclbin"),
        insts_path=os.path.join(args.out, "insts.bin"),
    )
    print(f"built {xclbin} {insts}")


if __name__ == "__main__":
    main()
