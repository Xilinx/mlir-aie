# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

# RUN: export BASENAME=$(basename %s)
# RUN: rm -rf $BASENAME && mkdir $BASENAME && cd $BASENAME
# RUN: VITIS_DIR=$VITIS WORKDIR=$PWD XRT_DIR=%XRT_DIR %PYTHON %s

import sys

import numpy as np
from aie.extras.dialects.ext import arith, func, linalg
from aie.extras.runtime.passes import run_pipeline, Pipeline
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
    another_bd,
    bd_dim_layout,
)
from aie.dialects.aiex import ipu_sync
from aie.dialects.linalg.opdsl.ops.core_named_ops import fill
from aie.dialects.scf import for_, yield_
from aie.util import tiling_calculator_n_tiles
from aie.xrt import XCLBin
from util import (
    construct_and_print_module,
    INPUT_WITH_ADDRESSES_PIPELINE,
    AIE_LOWER_TO_LLVM,
    chess_compile,
    make_core_elf,
    make_design_pdi,
    make_xclbin,
    setup_xclbin_firmware,
    CREATE_PATH_FINDER_FLOWS,
    DMA_TO_IPU,
    ipu_writebd_shimtile,
    ipu_write32,
    link_with_chess_intrinsic_wrapper,
    process_bd,
    forward_bd,
    hold_lock,
)

bd_dim_layout = lambda *args: bd_dim_layout(*args)
range_ = for_

DMA = WireBundle.DMA
S2MM = DMAChannelDir.S2MM
MM2S = DMAChannelDir.MM2S
Acquire = LockAction.Acquire
AcquireGreaterEqual = LockAction.AcquireGreaterEqual
Release = LockAction.Release


# CHECK-LABEL: tiled_nonsquare_tile_matrix_mult
@construct_and_print_module
def tiled_nonsquare_tile_matrix_mult(module):
    M = N = 32

    tile_rows_A, tile_cols_A = 2, 1
    tile_rows_B, tile_cols_B = 1, 2
    tile_rows_C, tile_cols_C = 2, 2

    tile_m_A, tile_n_A = M // tile_rows_A, N // tile_cols_A
    tile_m_B, tile_n_B = M // tile_rows_B, N // tile_cols_B
    tile_m_C, tile_n_C = M // tile_rows_C, N // tile_cols_C

    (
        _,
        _,
        (d1_size_A, d1_stride_A),
        (d0_size_A, d0_stride_A),
    ) = tiling_calculator_n_tiles(
        M, N, n_tile_rows=tile_rows_A, n_tile_cols=tile_cols_A
    )
    (
        _,
        _,
        (d1_size_B, d1_stride_B),
        (d0_size_B, d0_stride_B),
    ) = tiling_calculator_n_tiles(
        M, N, n_tile_rows=tile_rows_B, n_tile_cols=tile_cols_B
    )
    (
        _,
        _,
        (d1_size_C, d1_stride_C),
        (d0_size_C, d0_stride_C),
    ) = tiling_calculator_n_tiles(
        M, N, n_tile_rows=tile_rows_C, n_tile_cols=tile_cols_C
    )

    @device(AIEDevice.ipu)
    def ipu():
        tile_0_0 = tile(0, 0)
        tile_0_1 = tile(0, 1)
        tile_0_2 = tile(0, 2)

        # in
        buffer_0_2_a = aie.buffer(T.memref(tile_m_A, tile_n_A, T.i32()), tile_0_2)
        buffer_0_2_b = aie.buffer(T.memref(tile_m_B, tile_n_B, T.i32()), tile_0_2)
        # out
        buffer_0_2_c = aie.buffer(T.memref(tile_m_C, tile_n_C, T.i32()), tile_0_2)

        # input
        lock_0_1_read_in_a = aie.lock(tile_0_1, lock_id=0, init=1)
        lock_0_1_write_out_a = aie.lock(tile_0_1, lock_id=1, init=0)
        lock_0_1_read_in_b = aie.lock(tile_0_1, lock_id=2, init=1)
        lock_0_1_write_out_b = aie.lock(tile_0_1, lock_id=3, init=0)
        # output/returning
        lock_0_1_read_in_c = aie.lock(tile_0_1, lock_id=4, init=1)
        lock_0_1_write_out_c = aie.lock(tile_0_1, lock_id=5, init=0)

        lock_0_2_read_in_a = aie.lock(tile_0_2, lock_id=0, init=1)
        lock_0_2_use_a = aie.lock(tile_0_2, lock_id=1, init=0)
        lock_0_2_read_in_b = aie.lock(tile_0_2, lock_id=2, init=1)
        lock_0_2_use_b = aie.lock(tile_0_2, lock_id=3, init=0)
        lock_0_2_use_c = aie.lock(tile_0_2, lock_id=4, init=1)
        lock_0_2_write_out_c = aie.lock(tile_0_2, lock_id=5, init=0)

        # input flow
        # a
        aie.flow(tile_0_0, DMA, 0, tile_0_1, DMA, 0)
        aie.flow(tile_0_1, DMA, 0, tile_0_2, DMA, 0)
        # b
        aie.flow(tile_0_0, DMA, 1, tile_0_1, DMA, 1)
        aie.flow(tile_0_1, DMA, 1, tile_0_2, DMA, 1)
        # output flow
        aie.flow(tile_0_2, DMA, 0, tile_0_1, DMA, 2)
        aie.flow(tile_0_1, DMA, 2, tile_0_0, DMA, 0)

        @func.func(emit=True)
        def bobsyouruncle():
            # in A
            channel_index = 0
            col = 0
            ddr_id = 0
            offsets = [
                0,
                # A tiles are "fat" so need to offset by rows (i.e. d1 dim)
                0 + d1_size_A * d1_stride_A,
            ]
            for i, bd_id in enumerate(range(2)):
                ipu_writebd_shimtile(
                    bd_id,
                    buffer_length=tile_m_A * tile_n_A,
                    offset=offsets[i],
                    ddr_id=ddr_id,
                )
                ipu_write32(MM2S, channel_index, col, bd_id)

            # in B
            channel_index = 1
            col = 0
            ddr_id = 1
            for bd_id in range(bd_id + 1, bd_id + 1 + 4, 2):
                ipu_writebd_shimtile(
                    bd_id,
                    buffer_length=tile_m_B * tile_n_B,
                    offset=0,
                    ddr_id=ddr_id,
                    d1_size=d1_size_B,
                    d1_stride=d1_stride_B,
                    d0_size=d0_size_B,
                    d0_stride=d0_stride_B,
                )
                ipu_write32(MM2S, channel_index, col, bd_id)
                bd_id += 1
                # B tiles are "tall" so need to offset by cols (i.e. d0 dim)
                ipu_writebd_shimtile(
                    bd_id,
                    buffer_length=tile_m_B * tile_n_B,
                    offset=d0_size_B * d0_stride_B,
                    ddr_id=ddr_id,
                    d1_size=d1_size_B,
                    d1_stride=d1_stride_B,
                    d0_size=d0_size_B,
                    d0_stride=d0_stride_B,
                )
                ipu_write32(MM2S, channel_index, col, bd_id)

            # out C
            channel_index = 0
            col = 0
            ddr_id = 2
            offsets = [
                0,
                0 + d0_size_C * d0_stride_C,
                d1_size_C * d1_stride_C,
                d1_size_C * d1_stride_C + d0_size_C * d0_stride_C,
            ]

            for i, bd_id in enumerate(range(bd_id + 1, bd_id + 1 + 4)):
                ipu_writebd_shimtile(
                    bd_id,
                    buffer_length=tile_m_C * tile_n_C,
                    offset=offsets[i],
                    ddr_id=ddr_id,
                    d1_size=d1_size_C,
                    d1_stride=d1_stride_C,
                    d0_size=d0_size_C,
                    d0_stride=d0_stride_C,
                )
                ipu_write32(S2MM, channel_index, col, bd_id)
                ipu_sync(
                    channel=0, column=0, column_num=1, direction=0, row=0, row_num=1
                )

        @memtile_dma(tile_0_1)
        def memtile_dma_0_1():
            # input flow
            buffer_0_1_a = aie.buffer(T.memref(tile_m_A, tile_n_A, T.i32()), tile_0_1)
            buffer_0_1_b = aie.buffer(T.memref(tile_m_B, tile_n_B, T.i32()), tile_0_1)
            # output flow
            buffer_0_1_c = aie.buffer(T.memref(tile_m_C, tile_n_C, T.i32()), tile_0_1)

            @dma(S2MM, 0)
            def dma1():
                aie.use_lock(lock_0_1_read_in_a, AcquireGreaterEqual)
                aie.dma_bd(buffer_0_1_a)
                aie.use_lock(lock_0_1_write_out_a, Release)

            @dma(MM2S, 0, num_blocks=2)
            def dma2():
                aie.use_lock(lock_0_1_write_out_a, AcquireGreaterEqual)
                aie.dma_bd(buffer_0_1_a)
                aie.use_lock(lock_0_1_write_out_a, Release)

            @another_bd(dma2)
            def dma2point5():
                aie.use_lock(lock_0_1_write_out_a, AcquireGreaterEqual)
                aie.dma_bd(buffer_0_1_a)
                aie.use_lock(lock_0_1_read_in_a, Release)

            @dma(S2MM, 1)
            def dma3():
                aie.use_lock(lock_0_1_read_in_b, AcquireGreaterEqual)
                aie.dma_bd(buffer_0_1_b)
                aie.use_lock(lock_0_1_write_out_b, Release)

            @dma(MM2S, 1)
            def dma4():
                aie.use_lock(lock_0_1_write_out_b, AcquireGreaterEqual)
                aie.dma_bd(buffer_0_1_b)
                aie.use_lock(lock_0_1_read_in_b, Release)

            @dma(S2MM, 2)
            def dma5():
                aie.use_lock(lock_0_1_read_in_c, AcquireGreaterEqual)
                aie.dma_bd(buffer_0_1_c)
                aie.use_lock(lock_0_1_write_out_c, Release)

            @dma(MM2S, 2)
            def dma6():
                aie.use_lock(lock_0_1_write_out_c, AcquireGreaterEqual)
                aie.dma_bd(buffer_0_1_c)
                aie.use_lock(lock_0_1_read_in_c, Release)

            aie.end()

        @mem(tile_0_2)
        def mem_0_2():
            # input
            @dma(S2MM, 0)
            def dma1():
                aie.use_lock(lock_0_2_read_in_a, AcquireGreaterEqual)
                aie.dma_bd(buffer_0_2_a)
                aie.use_lock(lock_0_2_use_a, Release)

            @dma(S2MM, 1)
            def dma2():
                aie.use_lock(lock_0_2_read_in_b, AcquireGreaterEqual)
                aie.dma_bd(buffer_0_2_b)
                aie.use_lock(lock_0_2_use_b, Release)

            # output
            @dma(MM2S, 0)
            def dma3():
                aie.use_lock(lock_0_2_write_out_c, AcquireGreaterEqual)
                aie.dma_bd(buffer_0_2_c)
                aie.use_lock(lock_0_2_use_c, Release)

            aie.end()

        @aie.core(tile_0_2)
        def core():
            for _ in range_(0, tile_rows_C):
                for _ in range_(0, tile_cols_C):
                    # wait on both in and out to be ready
                    # these have to be acge for some reason...
                    aie.use_lock(lock_0_2_use_a, AcquireGreaterEqual)
                    aie.use_lock(lock_0_2_use_b, AcquireGreaterEqual)
                    aie.use_lock(lock_0_2_use_c, AcquireGreaterEqual)

                    fill(arith.constant(0), outs=[buffer_0_2_c])
                    linalg.matmul(buffer_0_2_a, buffer_0_2_b, buffer_0_2_c)

                    aie.use_lock(lock_0_2_read_in_a, Release)
                    aie.use_lock(lock_0_2_read_in_b, Release)
                    aie.use_lock(lock_0_2_write_out_c, Release)
                    yield_([])
                yield_([])

    module = run_pipeline(module, Pipeline().canonicalize())
    lowered_linalg = run_pipeline(
        module, Pipeline().convert_linalg_to_loops().fold_memref_alias_ops()
    )
    input_with_addresses = run_pipeline(lowered_linalg, INPUT_WITH_ADDRESSES_PIPELINE)
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

    # _GlobalDebug.flag = True
    generate_cdo(input_physical.operation, str(util.WORKDIR))
    # _GlobalDebug.flag = False
    make_design_pdi()

    generated_ipu_insts = run_pipeline(input_with_addresses, DMA_TO_IPU)
    ipu_insts = [int(inst, 16) for inst in ipu_instgen(generated_ipu_insts.operation)]

    xclbin_path = make_xclbin(module)
    with FileLock("/tmp/ipu.lock"):
        setup_xclbin_firmware(xclbin_path)

        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        xclbin.load_ipu_instructions(ipu_insts)
        inps, outps = xclbin.mmap_buffers([(M, N), (M, N)], [(M, N)], np.int32)

        wrap_A = np.asarray(inps[0])
        wrap_B = np.asarray(inps[1])
        wrap_C = np.asarray(outps[0])

        A = np.random.randint(0, 10, (M, N), dtype=np.int32)
        B = np.random.randint(0, 10, (M, N), dtype=np.int32)
        C = np.zeros((M, N), dtype=np.int32)

        np.copyto(wrap_A, A, casting="no")
        np.copyto(wrap_B, B, casting="no")
        np.copyto(wrap_C, C, casting="no")

        xclbin.sync_buffers_to_device()
        xclbin.run()
        print("Running kernel")
        xclbin.wait(30)
        xclbin.sync_buffers_from_device()

        if not np.array_equal(A @ B, wrap_C):
            with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
                print(A @ B)
                print(wrap_C)


# CHECK-LABEL: tiled_nonsquare_tile_matrix_mult_sugar
@construct_and_print_module
def tiled_nonsquare_tile_matrix_mult_sugar(module):
    M = N = 32

    tile_rows_A, tile_cols_A = 2, 1
    tile_rows_B, tile_cols_B = 1, 2
    tile_rows_C, tile_cols_C = 2, 2

    tile_m_A, tile_n_A = M // tile_rows_A, N // tile_cols_A
    tile_m_B, tile_n_B = M // tile_rows_B, N // tile_cols_B
    tile_m_C, tile_n_C = M // tile_rows_C, N // tile_cols_C

    (
        _,
        _,
        (d1_size_A, d1_stride_A),
        (d0_size_A, d0_stride_A),
    ) = tiling_calculator_n_tiles(
        M, N, n_tile_rows=tile_rows_A, n_tile_cols=tile_cols_A
    )
    (
        _,
        _,
        (d1_size_B, d1_stride_B),
        (d0_size_B, d0_stride_B),
    ) = tiling_calculator_n_tiles(
        M, N, n_tile_rows=tile_rows_B, n_tile_cols=tile_cols_B
    )
    (
        _,
        _,
        (d1_size_C, d1_stride_C),
        (d0_size_C, d0_stride_C),
    ) = tiling_calculator_n_tiles(
        M, N, n_tile_rows=tile_rows_C, n_tile_cols=tile_cols_C
    )

    @device(AIEDevice.ipu)
    def ipu():
        tile_0_0 = tile(0, 0)
        tile_0_1 = tile(0, 1)
        tile_0_2 = tile(0, 2)

        # in
        buffer_0_2_a = aie.buffer(T.memref(tile_m_A, tile_n_A, T.i32()), tile_0_2)
        buffer_0_2_b = aie.buffer(T.memref(tile_m_B, tile_n_B, T.i32()), tile_0_2)
        # out
        buffer_0_2_c = aie.buffer(T.memref(tile_m_C, tile_n_C, T.i32()), tile_0_2)

        # input
        lock_0_1_read_in_a = aie.lock(tile_0_1, lock_id=0, init=1)
        lock_0_1_write_out_a = aie.lock(tile_0_1, lock_id=1, init=0)

        lock_0_2_read_in_a = aie.lock(tile_0_2, lock_id=0, init=1)
        lock_0_2_use_a = aie.lock(tile_0_2, lock_id=1, init=0)
        lock_0_2_read_in_b = aie.lock(tile_0_2, lock_id=2, init=1)
        lock_0_2_use_b = aie.lock(tile_0_2, lock_id=3, init=0)
        lock_0_2_use_c = aie.lock(tile_0_2, lock_id=4, init=1)
        lock_0_2_write_out_c = aie.lock(tile_0_2, lock_id=5, init=0)

        # input flow
        # a
        aie.flow(tile_0_0, DMA, 0, tile_0_1, DMA, 0)
        aie.flow(tile_0_1, DMA, 0, tile_0_2, DMA, 0)
        # b
        aie.flow(tile_0_0, DMA, 1, tile_0_1, DMA, 1)
        aie.flow(tile_0_1, DMA, 1, tile_0_2, DMA, 1)
        # output flow
        aie.flow(tile_0_2, DMA, 0, tile_0_1, DMA, 2)
        aie.flow(tile_0_1, DMA, 2, tile_0_0, DMA, 0)

        @func.func(emit=True)
        def bobsyouruncle():
            # in A
            channel_index = 0
            col = 0
            ddr_id = 0
            offsets = [
                0,
                # A tiles are "fat" so need to offset by rows (i.e. d1 dim)
                0 + d1_size_A * d1_stride_A,
            ]
            for i, bd_id in enumerate(range(2)):
                ipu_writebd_shimtile(
                    bd_id,
                    buffer_length=tile_m_A * tile_n_A,
                    offset=offsets[i],
                    ddr_id=ddr_id,
                )
                ipu_write32(MM2S, channel_index, col, bd_id)

            # in B
            channel_index = 1
            col = 0
            ddr_id = 1
            for bd_id in range(bd_id + 1, bd_id + 1 + 4, 2):
                ipu_writebd_shimtile(
                    bd_id,
                    buffer_length=tile_m_B * tile_n_B,
                    offset=0,
                    ddr_id=ddr_id,
                    d1_size=d1_size_B,
                    d1_stride=d1_stride_B,
                    d0_size=d0_size_B,
                    d0_stride=d0_stride_B,
                )
                ipu_write32(MM2S, channel_index, col, bd_id)
                bd_id += 1
                # B tiles are "tall" so need to offset by cols (i.e. d0 dim)
                ipu_writebd_shimtile(
                    bd_id,
                    buffer_length=tile_m_B * tile_n_B,
                    offset=d0_size_B * d0_stride_B,
                    ddr_id=ddr_id,
                    d1_size=d1_size_B,
                    d1_stride=d1_stride_B,
                    d0_size=d0_size_B,
                    d0_stride=d0_stride_B,
                )
                ipu_write32(MM2S, channel_index, col, bd_id)

            # out C
            channel_index = 0
            col = 0
            ddr_id = 2
            offsets = [
                0,
                0 + d0_size_C * d0_stride_C,
                d1_size_C * d1_stride_C,
                d1_size_C * d1_stride_C + d0_size_C * d0_stride_C,
            ]

            for i, bd_id in enumerate(range(bd_id + 1, bd_id + 1 + 4)):
                ipu_writebd_shimtile(
                    bd_id,
                    buffer_length=tile_m_C * tile_n_C,
                    offset=offsets[i],
                    ddr_id=ddr_id,
                    d1_size=d1_size_C,
                    d1_stride=d1_stride_C,
                    d0_size=d0_size_C,
                    d0_stride=d0_stride_C,
                )
                ipu_write32(S2MM, channel_index, col, bd_id)
                ipu_sync(
                    channel=0, column=0, column_num=1, direction=0, row=0, row_num=1
                )

        @memtile_dma(tile_0_1)
        def memtile_dma_0_1():
            # input flow
            buffer_0_1_a = aie.buffer(T.memref(tile_m_A, tile_n_A, T.i32()), tile_0_1)
            buffer_0_1_b = aie.buffer(T.memref(tile_m_B, tile_n_B, T.i32()), tile_0_1)
            # output flow
            buffer_0_1_c = aie.buffer(T.memref(tile_m_C, tile_n_C, T.i32()), tile_0_1)

            @dma(S2MM, 0)
            def dma1():
                process_bd(lock_0_1_read_in_a, buffer_0_1_a, lock_0_1_write_out_a)

            @dma(MM2S, 0, num_blocks=2)
            def dma2():
                process_bd(lock_0_1_write_out_a, buffer_0_1_a, lock_0_1_write_out_a)

            @another_bd(dma2)
            def dma2point5():
                process_bd(lock_0_1_write_out_a, buffer_0_1_a, lock_0_1_read_in_a)

            forward_bd(tile_0_1, 1, buffer_0_1_b)
            forward_bd(tile_0_1, 2, buffer_0_1_c)

            aie.end()

        @mem(tile_0_2)
        def mem_0_2():
            # input
            @dma(S2MM, 0)
            def dma1():
                process_bd(lock_0_2_read_in_a, buffer_0_2_a, lock_0_2_use_a)

            @dma(S2MM, 1)
            def dma2():
                process_bd(lock_0_2_read_in_b, buffer_0_2_b, lock_0_2_use_b)

            # output
            @dma(MM2S, 0)
            def dma3():
                process_bd(lock_0_2_write_out_c, buffer_0_2_c, lock_0_2_use_c)

            aie.end()

        @aie.core(tile_0_2)
        def core():
            for _ in range_(0, tile_rows_C):
                for _ in range_(0, tile_cols_C):
                    with (
                        hold_lock(lock_0_2_use_a, lock_0_2_read_in_a),
                        hold_lock(lock_0_2_use_b, lock_0_2_read_in_b),
                        hold_lock(lock_0_2_use_c, lock_0_2_write_out_c),
                    ):
                        fill(arith.constant(0), outs=[buffer_0_2_c])
                        linalg.matmul(buffer_0_2_a, buffer_0_2_b, buffer_0_2_c)
                    yield_([])
                yield_([])

    module = run_pipeline(module, Pipeline().canonicalize())
    lowered_linalg = run_pipeline(
        module, Pipeline().convert_linalg_to_loops().fold_memref_alias_ops()
    )
    input_with_addresses = run_pipeline(lowered_linalg, INPUT_WITH_ADDRESSES_PIPELINE)
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

    # _GlobalDebug.flag = True
    generate_cdo(input_physical.operation, str(util.WORKDIR))
    # _GlobalDebug.flag = False
    make_design_pdi()

    generated_ipu_insts = run_pipeline(input_with_addresses, DMA_TO_IPU)
    ipu_insts = [int(inst, 16) for inst in ipu_instgen(generated_ipu_insts.operation)]

    xclbin_path = make_xclbin(module)
    with FileLock("/tmp/ipu.lock"):
        setup_xclbin_firmware(xclbin_path)

        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        xclbin.load_ipu_instructions(ipu_insts)
        inps, outps = xclbin.mmap_buffers([(M, N), (M, N)], [(M, N)], np.int32)

        wrap_A = np.asarray(inps[0])
        wrap_B = np.asarray(inps[1])
        wrap_C = np.asarray(outps[0])

        A = np.random.randint(0, 10, (M, N), dtype=np.int32)
        B = np.random.randint(0, 10, (M, N), dtype=np.int32)
        C = np.zeros((M, N), dtype=np.int32)

        np.copyto(wrap_A, A, casting="no")
        np.copyto(wrap_B, B, casting="no")
        np.copyto(wrap_C, C, casting="no")

        xclbin.sync_buffers_to_device()
        xclbin.run()
        print("Running kernel")
        xclbin.wait(30)
        xclbin.sync_buffers_from_device()

        if not np.array_equal(A @ B, wrap_C):
            with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
                print(A @ B)
                print(wrap_C)
