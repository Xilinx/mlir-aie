# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

# RUN: export BASENAME=$(basename %s)
# RUN: rm -rf $BASENAME && mkdir $BASENAME && cd $BASENAME
# RUN: VITIS_DIR=$VITIS WORKDIR=$PWD XRT_DIR=%XRT_DIR %PYTHON %s

import random

import numpy as np
from aie.extras.dialects.ext import memref, arith, func
from aie.extras.runtime.passes import run_pipeline
from filelock import FileLock

import aie.extras.types as T
import util
from aie.compiler.aiecc.main import (
    generate_cores_list,
)
from aie.dialects import aie
from aie.dialects.aie import (
    AIEDevice,
    DMAChannelDir,
    LockAction,
    WireBundle,
    device,
    generate_bcf,
    generate_cdo,
    ipu_instgen,
    mem,
    memtile_dma,
    tile,
    translate_mlir_to_llvmir,
    dma,
)
from aie.dialects.aiex import ipu_sync, ipu_dma_memcpy_nd
from aie.dialects.scf import for_
from aie.dialects.scf import yield_
from aie.xrt import XCLBin
from util import (
    construct_and_print_module,
    chess_compile,
    make_core_elf,
    make_design_pdi,
    make_xclbin,
    setup_xclbin_firmware,
    link_with_chess_intrinsic_wrapper,
)

from aie.compiler.aiecc.main import (
    INPUT_WITH_ADDRESSES_PIPELINE,
    AIE_LOWER_TO_LLVM,
    CREATE_PATH_FINDER_FLOWS,
    DMA_TO_IPU,
)

range_ = for_

DMA = WireBundle.DMA
S2MM = DMAChannelDir.S2MM
MM2S = DMAChannelDir.MM2S
Acquire = LockAction.Acquire
AcquireGreaterEqual = LockAction.AcquireGreaterEqual
Release = LockAction.Release


# CHECK-LABEL: add_256_using_dma_op_no_double_buffering
@construct_and_print_module
def add_256_using_dma_op_no_double_buffering(module):
    RANDOM_NUMBER = random.randint(0, 100)
    LEN = 128
    LOCAL_MEM_SIZE = 32

    @device(AIEDevice.ipu)
    def ipu():
        tile_0_0 = tile(0, 0)
        tile_0_1 = tile(0, 1)
        tile_0_2 = tile(0, 2)

        # in
        buffer_0_2 = aie.buffer(T.memref(LOCAL_MEM_SIZE, T.i32()), tile_0_2)
        # out
        buffer_0_2_1 = aie.buffer(T.memref(LOCAL_MEM_SIZE, T.i32()), tile_0_2)

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
            random_number = arith.constant(RANDOM_NUMBER)
            for _ in range_(0, LEN // LOCAL_MEM_SIZE):
                # wait on both in and out to be ready
                # these have to be acge for some reason...
                aie.use_lock(lock_0_2_1, AcquireGreaterEqual)
                aie.use_lock(lock_0_2_2, AcquireGreaterEqual)

                for arg1 in range_(0, LOCAL_MEM_SIZE):
                    v0 = memref.load(buffer_0_2, [arg1])
                    v1 = arith.addi(v0, random_number)
                    memref.store(v1, buffer_0_2_1, [arg1])
                    yield_([])

                aie.use_lock(lock_0_2_0, Release)
                aie.use_lock(lock_0_2_3, Release)

                yield_([])

        # this is gibberish - everything from here to the end of "bobsyouruncle"
        this_is_meaningless_1 = memref.global_(
            "this_is_meaningless_1",
            T.memref(1, T.f8E4M3B11FNUZ()),
            sym_visibility="public",
        ).opview
        this_is_meaningless_2 = memref.global_(
            "this_is_meaningless_2",
            T.memref(1, T.f8E4M3B11FNUZ()),
            sym_visibility="public",
        ).opview
        aie.shim_dma_allocation(this_is_meaningless_1.sym_name.value, MM2S, 0, 0)
        aie.shim_dma_allocation(this_is_meaningless_2.sym_name.value, S2MM, 0, 0)

        @func.func(emit=True)
        def bobsyouruncle(
            arg0: T.memref(LEN, T.i32()),
            _arg1: T.memref(1, T.i32()),
            arg2: T.memref(LEN, T.i32()),
        ):
            ipu_dma_memcpy_nd(
                this_is_meaningless_1.sym_name.value,
                0,
                arg0,
                [0, 0, 0, 0],
                [1, 1, 1, LEN],
                [0, 0, 0],
            )
            ipu_dma_memcpy_nd(
                this_is_meaningless_2.sym_name.value,
                1,
                arg2,
                [0, 0, 0, 0],
                [1, 1, 1, LEN],
                [0, 0, 0],
            )

            ipu_sync(channel=0, column=0, column_num=1, direction=0, row=0, row_num=1)

        @memtile_dma(tile_0_1)
        def memtile_dma_0_1():
            # input flow
            buffer_0_1 = aie.buffer(T.memref(LOCAL_MEM_SIZE, T.i32()), tile_0_1)
            # output flow
            buffer_0_1_0 = aie.buffer(T.memref(LOCAL_MEM_SIZE, T.i32()), tile_0_1)

            @dma(S2MM, 0)
            def dma1():
                aie.use_lock(lock_0_1_0, AcquireGreaterEqual)
                aie.dma_bd(buffer_0_1)
                aie.use_lock(lock_0_1_1, Release)

            @dma(MM2S, 0)
            def dma2():
                aie.use_lock(lock_0_1_1, AcquireGreaterEqual)
                aie.dma_bd(buffer_0_1)
                aie.use_lock(lock_0_1_0, Release)

            @dma(S2MM, 1)
            def dma3():
                aie.use_lock(lock_0_1_2, AcquireGreaterEqual)
                aie.dma_bd(buffer_0_1_0)
                aie.use_lock(lock_0_1_3, Release)

            @dma(MM2S, 1)
            def dma4():
                aie.use_lock(lock_0_1_3, AcquireGreaterEqual)
                aie.dma_bd(buffer_0_1_0)
                aie.use_lock(lock_0_1_2, Release)

            aie.end()

        @mem(tile_0_2)
        def mem_0_2():
            # input
            @dma(S2MM, 0)
            def dma1():
                aie.use_lock(lock_0_2_0, AcquireGreaterEqual)
                aie.dma_bd(buffer_0_2)
                aie.use_lock(lock_0_2_1, Release)

            # output
            @dma(MM2S, 0)
            def dma2():
                aie.use_lock(lock_0_2_3, AcquireGreaterEqual)
                aie.dma_bd(buffer_0_2_1)
                aie.use_lock(lock_0_2_2, Release)

            aie.end()

    input_with_addresses = run_pipeline(module, INPUT_WITH_ADDRESSES_PIPELINE)
    input_opt_with_addresses = run_pipeline(input_with_addresses, AIE_LOWER_TO_LLVM)
    chess_compile(
        link_with_chess_intrinsic_wrapper(
            translate_mlir_to_llvmir(input_opt_with_addresses.operation)
        )
    )

    [(col, row, _)] = generate_cores_list(str(input_with_addresses))
    core_bcf = generate_bcf(input_with_addresses.operation, col, row)
    make_core_elf(core_bcf)

    input_physical = run_pipeline(input_with_addresses, CREATE_PATH_FINDER_FLOWS)

    generate_cdo(input_physical.operation, str(util.WORKDIR))
    make_design_pdi()

    generated_ipu_insts = run_pipeline(input_with_addresses, DMA_TO_IPU)
    ipu_insts = [int(inst, 16) for inst in ipu_instgen(generated_ipu_insts.operation)]

    xclbin_path = make_xclbin(module)
    with FileLock("/tmp/ipu.lock"):
        setup_xclbin_firmware(xclbin_path)

        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        xclbin.load_ipu_instructions(ipu_insts)
        inps, outps = xclbin.mmap_buffers([(LEN,), (LEN,)], [(LEN,)], np.int32)

        wrap_A = np.asarray(inps[0])
        wrap_C = np.asarray(outps[0])

        A = np.random.randint(0, 10, LEN, dtype=np.int32)
        C = np.zeros(LEN, dtype=np.int32)

        np.copyto(wrap_A, A, casting="no")
        np.copyto(wrap_C, C, casting="no")

        xclbin.sync_buffers_to_device()
        xclbin.run()
        xclbin.wait()
        xclbin.sync_buffers_from_device()

        assert np.allclose(A + RANDOM_NUMBER, wrap_C)
