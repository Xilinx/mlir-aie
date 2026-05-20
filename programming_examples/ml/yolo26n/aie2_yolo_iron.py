"""Full-network IRON design for yolo26n-cls on NPU2 Strix Point.

Threads all 11 block builders from aie2_yolo_per_block.py end-to-end:

    DRAM -> shim_in(0,0) -> m0 -> m1 -> ... -> m9 -> m10 -> shim_out(7,0) -> DRAM

Each block builder returns (out_fifo, [workers]); this file just connects
block N's `out_fifo` as block N+1's `act_in` and aggregates all workers
into a single Runtime sequence.

Usage (on Linux with mlir-aie installed):

    python3 aie2_yolo_iron.py > yolo_iron.mlir

The chain processes ONE sample per Runtime invocation, matching mobilenet's
convention. For batch=4 throughput, the host invokes the design 4 times
back-to-back, filling the pipeline. See README § "Known limitations" for the
PSA batch-loop follow-up.

Inter-block shape coherence (manually verified vs yolo_spec.NETWORK):
    m0  out (256,256,16) -> m1 in (256,256,16)
    m1  out (128,128,32) -> m2 in (128,128,32)
    m2  out (128,128,64) -> m3 in (128,128,64)
    m3  out (64,64,64)   -> m4 in (64,64,64)
    m4  out (64,64,128)  -> m5 in (64,64,128)
    m5  out (32,32,128)  -> m6 in (32,32,128)
    m6  out (32,32,128)  -> m7 in (32,32,128)
    m7  out (16,16,256)  -> m8 in (16,16,256)
    m8  out (16,16,256)  -> m9 in (16,16,256)
    m9  out (16,16,256)  -> m10 in (16,16,256)
    m10 out (2,)         -> shim_out
"""

import os
import sys
import pathlib

import numpy as np

from aie.iron import ObjectFifo, Program, Runtime
from aie.iron.device import NPU2

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import yolo_spec
import placement
from aie2_yolo_per_block import _BUILDERS, _load_manifest, _i8, _i32


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------
def yolo_iron() -> str:
    """Build the full yolo26n-cls IRON design and return resolved MLIR text."""
    manifest = _load_manifest()

    # ------- Input shape -------
    # m0 takes 512x512 RGB. We pad to 8 input channels on the host (yolo_spec /
    # _build_m0 expect the runtime input to be 8-channel even though the
    # manifest's raw weights are 3-channel; see _build_m0 docstring).
    m0 = yolo_spec.block("m0")
    in_w, in_h, _ = m0.layers[0].in_shape
    IN_C_PADDED = 8

    in_bytes = in_w * in_h * IN_C_PADDED  # 2,097,152 per sample
    in_ty = _i32((in_bytes // 4,))  # 524,288 i32

    # ------- Output shape -------
    # m10 emits a flat (2,) vector of logits. Round to a 1-element i32 slot.
    m10 = yolo_spec.block("m10")
    out_elems = int(np.prod(m10.layers[-1].out_shape))  # 2
    out_ty = _i32(((out_elems + 3) // 4,))  # 1 i32

    # ------- Initial shim-input fifo -------
    # Element type: one row of int8 activations, padded to 8 channels for m0.
    act_in_fifo = ObjectFifo(_i8((in_w, 1, IN_C_PADDED)), depth=5)

    # ------- Chain the 11 block builders -------
    # Iterate in declaration order from yolo_spec.NETWORK (m0, m1, ..., m10).
    # Each builder returns (out_fifo, [workers]); thread out_fifo as the next
    # block's act_in.
    current_fifo = act_in_fifo
    all_workers = []
    for blk in yolo_spec.NETWORK:
        builder = _BUILDERS[blk.name]
        out_fifo, workers = builder(current_fifo, manifest)
        all_workers.extend(workers)
        current_fifo = out_fifo

    # `current_fifo` is now m10's out_fifo (2 logits per sample).
    final_out = current_fifo

    # ------- Runtime sequence -------
    rt = Runtime()
    with rt.sequence(in_ty, out_ty) as (inp, out):
        rt.start(*all_workers)
        tg = rt.task_group()
        rt.fill(
            act_in_fifo.prod(),
            inp,
            tile=placement.PLACEMENT["shim"]["input"],  # Tile(0, 0)
            task_group=tg,
        )
        rt.drain(
            final_out.cons(),
            out,
            wait=True,
            tile=placement.PLACEMENT["shim"]["output"],  # Tile(7, 0)
            task_group=tg,
        )
        rt.finish_task_group(tg)

    return Program(NPU2(), rt).resolve_program()


if __name__ == "__main__":
    print(yolo_iron())
