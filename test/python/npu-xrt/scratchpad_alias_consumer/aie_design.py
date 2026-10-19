# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# IRON design: one run writes another run's DMA offset through a scratchpad alias.
#
# Sequence "consume" moves 4 tiles of 8 values from value `start` on, where
# `start` is a scratchpad parameter used only as a DMA offset (addr kind, stored
# raw), and the core adds one to each value. Sequence "produce" copies one
# 32-bit word from its input to its output through the mem tile: bound to the
# consumer's scratchpad alias, its output is `start`.
#
# Usage:
#   python3 aie_design.py > aie.mlir

import numpy as np
from aie.dialects.aiex import npu_load_pdi
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import DeviceConfiguration, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2Col1
from aie.iron.scratchpad_parameter import ScratchpadParameter

N = 256
TILE = 8
TILES = 4


def design() -> str:
    buf_ty = np.ndarray[(N,), np.dtype[np.int32]]
    tile_ty = np.ndarray[(TILE,), np.dtype[np.int32]]
    word_ty = np.ndarray[(1,), np.dtype[np.int32]]
    start = ScratchpadParameter("start", np.int32)
    of_in = ObjectFifo(tile_ty, name="objfifo_in")
    of_out = ObjectFifo(tile_ty, name="objfifo_out")
    of_word = ObjectFifo(word_ty, name="objfifo_word")
    of_word_back = of_word.cons().forward(name="objfifo_word_back")

    def core_fn(of_in, of_out):
        for _ in range_(TILES):
            in_elem = of_in.acquire(1)
            out_elem = of_out.acquire(1)
            for i in range_(TILE):
                out_elem[i] = in_elem[i] + 1
            of_in.release(1)
            of_out.release(1)

    worker = Worker(core_fn, [of_in.cons(), of_out.prod()], while_true=False)

    def consume(in_tensor, out_tensor, in_h, out_h):
        npu_load_pdi(device_ref="empty")
        npu_load_pdi(device_ref="test")
        moved = TensorAccessPattern((N,), 0, [1, 1, 1, TILES * TILE], [0, 0, 0, 1])
        in_h.fill(in_tensor, tap=moved, offset_parameter=start)
        out_h.drain(out_tensor, tap=moved, wait=True)

    def produce(word, scratchpad, word_h, back_h):
        npu_load_pdi(device_ref="empty")
        npu_load_pdi(device_ref="test")
        word_h.fill(word)
        back_h.drain(scratchpad, wait=True)

    consumer = Runtime(
        consume,
        [buf_ty, buf_ty, of_in.prod(), of_out.cons()],
        name="consume",
        implicit_configure=False,
    )
    producer = Runtime(
        produce,
        [word_ty, word_ty, of_word.prod(), of_word_back.cons()],
        name="produce",
        implicit_configure=False,
    )
    config = DeviceConfiguration(
        "test", NPU2Col1(), workers=[worker], runtimes=[consumer, producer]
    )
    module = Program.compose([config], entry=consumer).resolve_program()
    # The firmware skips loading a PDI it loaded last, so each sequence loads
    # an empty device first to reload "test" and its parameters. The module
    # then holds two devices, which compose's count of one would contradict.
    for name in ("iron.entry", "iron.configuration_count"):
        del module.operation.attributes[name]
    empty = "module {\n  aie.device(npu2) @empty { }\n"
    return str(module).replace("module {\n", empty, 1)


print(design())
