# npu-xrt/e2e/test_add_256_using_dma_op_no_double_buffering.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates

import numpy as np
from pathlib import Path
import pytest
import random

from aie.dialects import aie, aiex
from aie.dialects.aie import (
    AIEDevice,
    DMAChannelDir,
    LockAction,
    WireBundle,
)
from aie.helpers.dialects.ext.scf import _for as range_
from aie.extras.dialects.ext import memref

# noinspection PyUnresolvedReferences
from aie.extras.testing import MLIRContext, mlir_ctx as ctx
import aie.extras.types as T
from aie.compiler.aiecc.main import run as aiecc_run
from aie.utils.xrt import setup_aie, execute as execute_on_aie

from util import aiecc_args

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


DMA = WireBundle.DMA
S2MM = DMAChannelDir.S2MM
MM2S = DMAChannelDir.MM2S
Acquire = LockAction.Acquire
AcquireGreaterEqual = LockAction.AcquireGreaterEqual
Release = LockAction.Release


def test_add_256_using_dma_op_no_double_buffering(ctx: MLIRContext, workdir: Path):
    RANDOM_NUMBER = random.randint(0, 100)
    LEN = 128
    LOCAL_MEM_SIZE = 32

    @aie.device(AIEDevice.npu1_1col)
    def npu():
        tile_0_0 = aie.tile(0, 0)
        tile_0_1 = aie.tile(0, 1)
        tile_0_2 = aie.tile(0, 2)

        # in
        buffer_0_2 = aie.buffer(
            tile_0_2, np.ndarray[(LOCAL_MEM_SIZE,), np.dtype[np.int32]]
        )
        # out
        buffer_0_2_1 = aie.buffer(
            tile_0_2, np.ndarray[(LOCAL_MEM_SIZE,), np.dtype[np.int32]]
        )

        lock_0_1_0 = aie.lock(tile_0_1, lock_id=0, init=1)
        lock_0_1_1 = aie.lock(tile_0_1, lock_id=1, init=0)
        lock_0_1_2 = aie.lock(tile_0_1, lock_id=2, init=1)
        lock_0_1_3 = aie.lock(tile_0_1, lock_id=3, init=0)

        lock_0_2_0 = aie.lock(tile_0_2, lock_id=0, init=1)
        lock_0_2_1 = aie.lock(tile_0_2, lock_id=1, init=0)
        lock_0_2_2 = aie.lock(tile_0_2, lock_id=2, init=1)
        lock_0_2_3 = aie.lock(tile_0_2, lock_id=3, init=0)

        # input flow
        aie.flow(tile_0_0, DMA, 0, tile_0_1, DMA, 0)
        aie.flow(tile_0_1, DMA, 0, tile_0_2, DMA, 0)
        # output flow
        aie.flow(tile_0_2, DMA, 0, tile_0_1, DMA, 1)
        aie.flow(tile_0_1, DMA, 1, tile_0_0, DMA, 0)

        @aie.core(tile_0_2)
        def core():
            for _ in range_(0, LEN // LOCAL_MEM_SIZE):
                # wait on both in and out to be ready
                # these have to be acge for some reason...
                aie.use_lock(lock_0_2_1, AcquireGreaterEqual)
                aie.use_lock(lock_0_2_2, AcquireGreaterEqual)

                for arg1 in range_(0, LOCAL_MEM_SIZE):
                    buffer_0_2_1[arg1] = buffer_0_2[arg1] + RANDOM_NUMBER
                aie.use_lock(lock_0_2_0, Release)
                aie.use_lock(lock_0_2_3, Release)

        # this is gibberish - everything from here to the end of "bobsyouruncle"
        this_is_meaningless_1 = memref.global_(
            sym_name="this_is_meaningless_1",
            type_=T.memref(1, T.f8E4M3B11FNUZ()),
            sym_visibility="public",
        ).opview
        this_is_meaningless_2 = memref.global_(
            sym_name="this_is_meaningless_2",
            type_=T.memref(1, T.f8E4M3B11FNUZ()),
            sym_visibility="public",
        ).opview
        aie.shim_dma_allocation(this_is_meaningless_1.sym_name.value, MM2S, 0, 0)
        aie.shim_dma_allocation(this_is_meaningless_2.sym_name.value, S2MM, 0, 0)

        @aiex.runtime_sequence(
            np.ndarray[(LEN,), np.dtype[np.int32]],
            np.ndarray[(LEN,), np.dtype[np.int32]],
            np.ndarray[(LEN,), np.dtype[np.int32]],
        )
        def bobsyouruncle(arg0, _arg1, arg2):
            aiex.npu_dma_memcpy_nd(
                this_is_meaningless_1.sym_name.value,
                0,
                arg0,
                [0, 0, 0, 0],
                [1, 1, 1, LEN],
                [0, 0, 0, 1],
            )
            aiex.npu_dma_memcpy_nd(
                this_is_meaningless_2.sym_name.value,
                1,
                arg2,
                [0, 0, 0, 0],
                [1, 1, 1, LEN],
                [0, 0, 0, 1],
            )

            aiex.npu_sync(
                channel=0, column=0, column_num=1, direction=0, row=0, row_num=1
            )

        @aie.memtile_dma(tile_0_1)
        def memtile_dma_0_1():
            # input flow
            buffer_0_1 = aie.buffer(
                tile_0_1, np.ndarray[(LOCAL_MEM_SIZE,), np.dtype[np.int32]]
            )
            # output flow
            buffer_0_1_0 = aie.buffer(
                tile_0_1, np.ndarray[(LOCAL_MEM_SIZE,), np.dtype[np.int32]]
            )

            @aie.dma(S2MM, 0)
            def dma1():
                aie.use_lock(lock_0_1_0, AcquireGreaterEqual)
                aie.dma_bd(buffer_0_1)
                aie.use_lock(lock_0_1_1, Release)

            @aie.dma(MM2S, 0)
            def dma2():
                aie.use_lock(lock_0_1_1, AcquireGreaterEqual)
                aie.dma_bd(buffer_0_1)
                aie.use_lock(lock_0_1_0, Release)

            @aie.dma(S2MM, 1)
            def dma3():
                aie.use_lock(lock_0_1_2, AcquireGreaterEqual)
                aie.dma_bd(buffer_0_1_0)
                aie.use_lock(lock_0_1_3, Release)

            @aie.dma(MM2S, 1)
            def dma4():
                aie.use_lock(lock_0_1_3, AcquireGreaterEqual)
                aie.dma_bd(buffer_0_1_0)
                aie.use_lock(lock_0_1_2, Release)

            aie.end()

        @aie.mem(tile_0_2)
        def mem_0_2():
            # input
            @aie.dma(S2MM, 0)
            def dma1():
                aie.use_lock(lock_0_2_0, AcquireGreaterEqual)
                aie.dma_bd(buffer_0_2)
                aie.use_lock(lock_0_2_1, Release)

            # output
            @aie.dma(MM2S, 0)
            def dma2():
                aie.use_lock(lock_0_2_3, AcquireGreaterEqual)
                aie.dma_bd(buffer_0_2_1)
                aie.use_lock(lock_0_2_2, Release)

            aie.end()

    INSTS = "npu_insts.txt"
    XCLBIN = "final.xclbin"
    aiecc_run(ctx.module, aiecc_args(XCLBIN, INSTS))
    app = setup_aie(
        XCLBIN,
        INSTS,
        LEN,
        np.int32,
        LEN,
        np.int32,
        LEN,
        np.int32,
    )
    A = np.random.randint(0, 10, LEN, dtype=np.int32)
    B = np.random.randint(0, 10, LEN, dtype=np.int32)
    aie_output = execute_on_aie(app, A, B)
    assert np.allclose(A + RANDOM_NUMBER, aie_output)
