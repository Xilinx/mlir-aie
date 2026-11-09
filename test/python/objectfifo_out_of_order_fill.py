# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

import numpy as np
import pytest

from aie.dialects import func
from aie.extras import types as T
from aie.extras.context import mlir_mod_ctx
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import ObjectFifo, Packet, Program, Runtime, Transport, Worker
from aie.iron.device import NPU2Col1
from aie.iron.runtime.data import RuntimeData
from aie.iron.scratchpad_parameter import ScratchpadParameter


def fifo(consumer_obj_type=None):
    return ObjectFifo(
        np.ndarray[(16,), np.dtype[np.int32]],
        name="input",
        consumer_obj_type=consumer_obj_type,
        transport=Transport.dma(packet=Packet(id=0), out_of_order=True),
    )


@pytest.mark.parametrize(
    "size,dtype,tap,packet_bytes",
    [
        (64, np.int32, None, 256),
        (64, np.int32, TensorAccessPattern((64,), 0, [8], [1]), 32),
        (64, np.int32, TensorAccessPattern((64,), 0, [4, 16], [16, 1]), 256),
        (16, np.int8, None, 16),
    ],
)
def test_reject_incorrect_packet_length(size, dtype, tap, packet_bytes):
    source = RuntimeData(np.ndarray[(size,), np.dtype[dtype]])
    with pytest.raises(
        ValueError,
        match=f"sends {packet_bytes} bytes per packet, but one receiving object requires 64 bytes",
    ):
        fifo().prod().fill(source, tap=tap, out_of_order_id=0)


def test_check_consumer_object_size():
    source = RuntimeData(np.ndarray[(16,), np.dtype[np.int32]])
    with pytest.raises(ValueError, match="one receiving object requires 128 bytes"):
        fifo(np.ndarray[(32,), np.dtype[np.int32]]).prod().fill(
            source, out_of_order_id=0
        )


def test_reject_runtime_length_parameter():
    source = RuntimeData(np.ndarray[(16,), np.dtype[np.int32]])
    with pytest.raises(ValueError, match="does not support length_parameter"):
        fifo().prod().fill(
            source,
            out_of_order_id=0,
            length_parameter=ScratchpadParameter("length", np.int32),
        )


def test_reject_runtime_packet_size():
    with mlir_mod_ctx():

        @func.FuncOp.from_py_func(T.index())
        def dynamic_size(n):
            tap = TensorAccessPattern((64,), 0, [1, 1, 1, n], [0, 0, 0, 1])
            source = RuntimeData(np.ndarray[(64,), np.dtype[np.int32]])
            with pytest.raises(ValueError, match="requires a static packet length"):
                fifo().prod().fill(source, tap=tap, out_of_order_id=0)


@pytest.mark.parametrize(
    "size,dtype,tap,repeats",
    [
        (16, np.int32, None, 1),
        (64, np.int8, None, 1),
        (64, np.int32, TensorAccessPattern((64,), 16, [16], [1]), 1),
        (64, np.int32, TensorAccessPattern((64,), 0, [4, 1, 1, 16], [16, 0, 0, 1]), 4),
        (16, np.int32, TensorAccessPattern((16,), 0, [4, 16], [0, 1]), 4),
    ],
)
def test_emit_object_sized_packets(size, dtype, tap, repeats):
    of_in = fifo()

    def consume(handle):
        handle.acquire(1)
        handle.release(1)

    worker = Worker(consume, [of_in.cons()])

    def sequence(source, handle):
        handle.fill(source, tap=tap, out_of_order_id=3)

    rt = Runtime(sequence, [np.ndarray[(size,), np.dtype[dtype]], of_in.prod()])
    module = Program(NPU2Col1(), rt, workers=[worker]).resolve_program()
    text = str(module)
    assert "out_of_order_id = 3 : i32" in text
    assert f"len = {64 // np.dtype(dtype).itemsize}" in text
    if repeats > 1:
        assert f"repeat_count = {repeats - 1} : i32" in text
