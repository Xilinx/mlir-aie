# (c) Copyright 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# IRON design: generates the MLIR for the scratchpad parameter test.
#
# Core computes: output = foo * bar (bf16 parameters set at runtime).
#
# Usage:
#   python3 aie_design.py > aie.mlir

import numpy as np
from ml_dtypes import bfloat16

from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2Col1
from aie.iron.parameter import Parameter
from aie.dialects.aiex import npu_load_pdi
from aie.dialects.arith import ConstantOp, mulf
from aie.dialects.memref import store
from aie.ir import IndexType, IntegerAttr


def design():
    device_name = "test"

    # Output type: 2 x bf16
    # Note: We only calculate 1 bf16 value, but DMAs operate at 4-byte granularity so this is the smallest possible size
    out_ty = np.ndarray[(2,), np.dtype[bfloat16]]

    # Parameters
    foo = Parameter("foo", bfloat16)
    bar = Parameter("bar", bfloat16)

    # ObjectFIFO for output
    of_out = ObjectFifo(out_ty, name="objfifo_out")

    # Core function: read parameters, multiply, write to output
    def core_fn(of_out, foo, bar):
        val_foo = foo.read()
        val_bar = bar.read()

        elem = of_out.acquire(1)
        result = val_foo * val_bar
        elem[0] = result
        of_out.release(1)

    worker = Worker(
        core_fn,
        [of_out.prod(), foo, bar],
        while_true=False,
    )

    # Runtime sequence: load empty device first to force PDI reconfiguration
    rt = Runtime()
    with rt.sequence(out_ty) as out_tensor:
        rt.inline_ops(lambda: npu_load_pdi(device_ref="empty"), [])
        rt.inline_ops(lambda: npu_load_pdi(device_ref=device_name), [])
        rt.sync_parameters()
        rt.start(worker)
        rt.drain(of_out.cons(), out_tensor, wait=True)

    module = Program(NPU2Col1(), rt).resolve_program(device_name=device_name)

    # Insert empty device at the beginning of the module to force PDI reload.
    # The firmware skips reloading a PDI if it's the same as the last one loaded,
    # so we force a different (empty) PDI first.
    # FIXME: Replace this with the proper IRON abstraction for multiple devices when it becomes available.
    mlir_text = str(module)
    empty_device = "  aie.device(npu2) @empty { }\n"
    mlir_text = mlir_text.replace("module {\n", "module {\n" + empty_device, 1)
    return mlir_text


mlir_text = design()
print(mlir_text)
