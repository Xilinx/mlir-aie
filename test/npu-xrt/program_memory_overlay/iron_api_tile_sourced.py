# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A slot written by a neighboring compute tile's own DMA, not the host.

Direct counterpart to test/npu-xrt/tile_sourced_ctrl_pkt_spike/aie.mlir, at
real ProgramMemorySlot scale instead of a 2-word proof: a source Worker's
core_fn calls `slot.load(overlay)` once per phase, and the target Worker's
`wait()` polls the plain flag Buffer the source's control-packet burst sets
on completion (there is no runtime sequence in this transport to release a
host-driven barrier).

Slot size (0x1FF0, just under this device's 0x2000 program-memory write
granule -- the largest a single-slot layout can place) is no longer capped
at a source tile's 16-entry BD table (`AIETargetModel::getNumBDs()`):
`_load_tile_sourced` sends every control-packet chunk through one reused,
self-looping `aie.dma_bd` (`next="self"`) instead of one static BD per
chunk. `BdIteration` was the obvious way to get this and is hardware-verified
to corrupt the packet's embedded address after the first repeat; a
self-looping BD is a different mechanism (an ordinary chain traversal, a
real descriptor fetch every hop) and is hardware-verified correct at this
size -- see ProgramMemorySlot._load_tile_sourced's comments.

Multiple `--tags` exercise multi-phase scheduling: `slot.load()` called more
than once, one overlay per phase, exactly like the host-written and
ping-pong transports support. Between phases, `_ensure_ack_rig`'s reverse
(destination -> source) DMA channel tells the source when it is actually
safe to overwrite the slot again -- the destination core reaching `wait()`
for phase N+1 is proof it has finished executing phase N's overlay.
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
SLOT_SIZE = 0x1FF0


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
        workload.compile_overlay_of_size(
            tag, N_ELEMS, out_o, SLOT_SIZE, entry="overlay_entry_a", workdir=args.out
        )
        object_files.append(out_o)

    set_current_device(from_name("npu2", n_cols=2))

    tile_ty = np.ndarray[(N_ELEMS,), np.dtype[np.int32]]
    host_in_ty = np.ndarray[(N_ELEMS,), np.dtype[np.int32]]
    host_out_shape = (len(tags), N_ELEMS)
    host_out_ty = np.ndarray[host_out_shape, np.dtype[np.int32]]

    def make_design():
        target_tile = Tile(0, 2)
        source_tile = Tile(1, 2)

        def source_core_fn():
            for overlay in overlays:
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
        overlays = [
            ProgramMemoryOverlay(f"w{tag}", slot, object_file_name=obj)
            for tag, obj in zip(tags, object_files)
        ]

        of_in = ObjectFifo(tile_ty, name="in")
        of_out = ObjectFifo(tile_ty, name="out")

        def target_core_fn(in_cons, out_prod, slot):
            for _ in range(len(tags)):
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
            for phase in range(len(tags)):
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
        program = Program(
            from_name("npu2", n_cols=2), rt, workers=[target_worker, source_worker]
        )
        return program, overlays

    xclbin, insts = ProgramMemoryOverlayDesign(make_design).compile(
        work_dir=args.out,
        xclbin_path=os.path.join(args.out, "aie.xclbin"),
        insts_path=os.path.join(args.out, "insts.bin"),
    )
    print(f"built {xclbin} {insts}")


if __name__ == "__main__":
    main()
