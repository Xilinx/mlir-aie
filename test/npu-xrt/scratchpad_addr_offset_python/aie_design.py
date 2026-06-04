# (c) Copyright 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# IRON design: DMA address offset patching via offset_parameter.
#
# REQUIRES: dont_run
# RUN: echo
#
# The host prepares an input buffer with monotonically increasing i32 values
# [0, 1, 2, ..., 31].  The core does a simple passthrough of 8 elements.
# The offset_parameter @input_offset controls where in the input buffer
# the DMA begins reading.
#
#   Run 1: input_offset = 0  -> output = [0, 1, 2, 3, 4, 5, 6, 7]
#   Run 2: input_offset = 8  -> output = [8, 9, 10, 11, 12, 13, 14, 15]
#   Run 3: input_offset = 16 -> output = [16, 17, 18, 19, 20, 21, 22, 23]
#
# Usage:
#   python3 aie_design.py > aie.mlir

import numpy as np

from aie.iron import ObjectFifo, Program, Runtime, Worker, WorkerRuntimeBarrier
from aie.iron.device import NPU2Col1
from aie.iron.parameter import Parameter
from aie.dialects.aiex import npu_load_pdi
from aie.helpers.taplib import TensorAccessPattern


def design():
    device_name = "test"

    # Types: input is 32 x i32 (full buffer), but each transfer is 8 elements
    in_ty = np.ndarray[(32,), np.dtype[np.int32]]
    out_ty = np.ndarray[(8,), np.dtype[np.int32]]
    tile_ty = np.ndarray[(8,), np.dtype[np.int32]]

    # Parameter: element offset into the input buffer
    input_offset = Parameter("input_offset", np.int32)

    # ObjectFIFOs
    of_in = ObjectFifo(tile_ty, name="objfifo_in")
    of_out = ObjectFifo(tile_ty, name="objfifo_out")

    # Barrier to gate the core until parameters are loaded
    barrier = WorkerRuntimeBarrier()

    # Core function: passthrough — copy input to output
    def core_fn(of_in, of_out, barrier):
        barrier.wait_for_value(1)
        barrier.release_with_value(0)

        in_elem = of_in.acquire(1)
        out_elem = of_out.acquire(1)
        for i in range(8):
            out_elem[i] = in_elem[i]
        of_in.release(1)
        of_out.release(1)

    worker = Worker(
        core_fn,
        [of_in.cons(), of_out.prod(), barrier],
        while_true=False,
    )

    # Runtime sequence
    rt = Runtime()
    with rt.sequence(in_ty, out_ty) as (in_tensor, out_tensor):
        rt.inline_ops(lambda: npu_load_pdi(device_ref="empty"), [])
        rt.inline_ops(lambda: npu_load_pdi(device_ref=device_name), [])
        rt.sync_parameters()
        rt.set_barrier(barrier, 1)
        rt.start(worker)

        # Input DMA — offset_parameter patches the BD address at runtime
        in_tap = TensorAccessPattern((32,), offset=0, sizes=[1, 1, 1, 8], strides=[0, 0, 0, 1])
        rt.fill(of_in.prod(), in_tensor, tap=in_tap, offset_parameter=input_offset)

        # Output DMA
        rt.drain(of_out.cons(), out_tensor, wait=True)

        rt.set_barrier(barrier, 0)

    module = Program(NPU2Col1(), rt).resolve_program(device_name=device_name)

    # Insert empty device to force PDI reload
    mlir_text = str(module)
    empty_device = '  aie.device(npu2) @empty { }\n'
    mlir_text = mlir_text.replace('module {\n', 'module {\n' + empty_device, 1)
    return mlir_text


mlir_text = design()
print(mlir_text)
