# test_algorithms_api.py -*- Python -*-
#
# Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""Unit tests for public IRON algorithm package exports."""

import numpy as np
import pytest

from aie.iron.device import NPU1Col1
from aie.utils import get_current_device
from aie.utils.hostruntime import set_current_device


def test_transform_export_is_callable_and_returns_module():
    from aie.iron.algorithms import transform

    assert callable(transform)

    set_current_device(NPU1Col1())
    try:
        tensor_ty = np.ndarray[(1024,), np.dtype[np.int32]]
        module = transform(lambda x: x + 1, tensor_ty, tile_size=16)
        assert module is not None
        assert hasattr(module, "operation")
    finally:
        set_current_device(None)


@pytest.mark.parametrize("iterations", [0, 1, 2])
def test_pipeline_stage_preserves_iteration_count(iterations):
    from aie.iron.algorithms._pipeline import Stage, pipeline
    from aie.iron.dataflow import ObjectFifo
    from aie.iron.kernel import Kernel

    previous_device = get_current_device(probe_runtime=False)
    set_current_device(NPU1Col1())
    try:
        tile_ty = np.ndarray[(16,), np.dtype[np.int32]]
        tensor_ty = np.ndarray[(max(iterations, 1) * 16,), np.dtype[np.int32]]
        source, sink = ObjectFifo(tile_ty, name="in"), ObjectFifo(tile_ty, name="out")
        step = Kernel("step", "step.o", arg_types=[tile_ty, tile_ty])
        stage = Stage(
            lambda ins, outs, held, constants, iteration: constants[0](ins[0], outs[0]),
            inputs=[(source, 1)],
            outputs=[sink],
            constants=[step],
            iterations=iterations,
        )
        module = pipeline(
            [stage],
            [tensor_ty, tensor_ty],
            [(source, "fill", 0), (sink, "drain", 1)],
        )
        core_ops = []

        def collect(op):
            if op.name == "aie.core":
                core_ops.append(str(op))
            for region in op.regions:
                for block in region.blocks:
                    for child in block.operations:
                        collect(child.operation)

        collect(module.operation)
        assert len(core_ops) == 1
        assert ("aie.objectfifo.acquire" in core_ops[0]) == (iterations > 0)
        assert ("aie.objectfifo.release" in core_ops[0]) == (iterations > 0)
        assert ("func.call @step" in core_ops[0]) == (iterations > 0)
    finally:
        set_current_device(previous_device)
