# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s
# RUN: %python %s | aie-opt --aie-place-tiles --aie-objectfifo-split \
# RUN:   --aie-objectfifo-allocate --aie-assign-buffer-addresses \
# RUN:   --aie-lower-scratchpad-parameters --aie-substitute-shim-dma-allocations \
# RUN:   --aie-assign-runtime-sequence-bd-ids --aie-dma-tasks-to-npu \
# RUN:   --aie-dma-to-npu | FileCheck %s --check-prefix=LOWER

# A mem tile task started with a repeat_parameter: up to 16 runs of a 64-value
# tile, as many as @reps + 1 says per call. Its push lowers, as a shim task's
# does, to a scratchpad update of the channel's task queue: S2MM 0 of mem tile
# (0, 1) at 0x1A0604, adding the host's count to issue_token | head BD 0.

import numpy as np

from aie.dialects.aie import BdIteration
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Bd, Buffer, Flow, Program, Runtime
from aie.iron.device import NPU2Col1, Tile
from aie.iron.scratchpad_parameter import ScratchpadParameter

N, TILE = 1024, 64

# CHECK: aiex.dma_configure_task_for @into_dst {
# CHECK: aie.dma_bd(%held : memref<1024xi32> len = 64)
# CHECK-SAME: iteration = #aie.bd_iteration<size = 16, stride = 64, current = 0>
# CHECK: } {issue_token = true, repeat_count = 15 : i32}
# CHECK: aiex.dma_start_task(%{{.*}}) {repeat_parameter = @reps}
# CHECK: aiex.dma_configure_task_for @into_src {
# CHECK: } {repeat_count = 15 : i32}
# CHECK: aiex.dma_start_task(%{{.*}}) {repeat_parameter = @reps}
# CHECK: aiex.scratchpad_parameter @reps : i32

# LOWER: aiex.npu.create_scratchpad {size = 4 : ui32}
# LOWER: aiex.npu.update_from_scratchpad {address = 1705476 : ui32, func_arg = 2147483648 : ui32, state_table_idx = 0 : ui8}
# LOWER: aiex.npu.update_from_scratchpad {address = 119316 : ui32, state_table_idx = 0 : ui8}
buf_ty = np.ndarray[(N,), np.dtype[np.int32]]
reps = ScratchpadParameter("reps", np.int32)
shim, mem = Tile(0, 0), Tile(0, 1)
held = Buffer(buf_ty, tile=mem, name="held")
into = Flow(shim, mem)


def sequence(in_tensor):
    stepped = TensorAccessPattern((N,), 0, [N // TILE, 1, 1, TILE], [TILE, 0, 0, 1])
    each = TensorAccessPattern((N,), 0, [TILE], [1])
    land = into.endpoint(mem).task(
        Bd(held, tap=each, iteration=BdIteration(N // TILE, TILE, 0)),
        runs=N // TILE,
        wait=True,
    )
    land.start(repeat_parameter=reps)
    into.fill(in_tensor, tap=stepped, repeat_parameter=reps)
    land.await_()


rt = Runtime(sequence, [buf_ty])
rt.add_flow(into)
print(Program(NPU2Col1(), rt).resolve_program())
