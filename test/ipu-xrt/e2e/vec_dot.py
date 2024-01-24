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

# this is to get the MemRefValue caster inside of aie-python-extras
# noinspection PyUnresolvedReferences
from aie.extras.dialects.ext import memref
from aie.extras.runtime.passes import run_pipeline, Pipeline
from filelock import FileLock

import aie.extras.types as T
import util
from aie._mlir_libs._mlir.ir import MemRefType
from aie.compiler.aiecc.main import (
    generate_cores_list,
)
from aie.dialects import aie, memref as memref_dialect
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
from aie.dialects.aiex import ipu_sync
from aie.dialects.linalg.opdsl.ops.core_named_ops import fill
from aie.dialects.scf import for_, yield_
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
    hold_lock,
    forward_bd,
    process_bd,
)

range_ = for_

DMA = WireBundle.DMA
S2MM = DMAChannelDir.S2MM
MM2S = DMAChannelDir.MM2S
Acquire = LockAction.Acquire
AcquireGreaterEqual = LockAction.AcquireGreaterEqual
Release = LockAction.Release


# CHECK-LABEL: vec_dot
@construct_and_print_module
def vec_dot(module):
    K = 32
    tiles = 4
    k = K // tiles

    @device(AIEDevice.ipu)
    def ipu():
        tile_0_0 = tile(0, 0)
        tile_0_1 = tile(0, 1)
        tile_0_2 = tile(0, 2)

        # in
        buffer_0_2_a = aie.buffer(T.memref(k, T.i32()), tile_0_2)
        buffer_0_2_b = aie.buffer(T.memref(k, T.i32()), tile_0_2)
        # out
        buffer_0_2_c = aie.buffer(T.memref(1, T.i32()), tile_0_2)

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
            offsets = list(range(0, K, k))
            for i, bd_id in enumerate(range(tiles)):
                ipu_writebd_shimtile(
                    bd_id,
                    buffer_length=k,
                    offset=offsets[i],
                    ddr_id=ddr_id,
                )
                ipu_write32(MM2S, channel_index, col, bd_id)

            # in B
            channel_index = 1
            col = 0
            ddr_id = 1
            for i, bd_id in enumerate(range(bd_id + 1, bd_id + 1 + tiles)):
                ipu_writebd_shimtile(
                    bd_id,
                    buffer_length=k,
                    offset=offsets[i],
                    ddr_id=ddr_id,
                )
                ipu_write32(MM2S, channel_index, col, bd_id)

            # out C
            channel_index = 0
            col = 0
            ddr_id = 2
            for i, bd_id in enumerate(range(bd_id + 1, bd_id + 1 + tiles)):
                ipu_writebd_shimtile(
                    bd_id,
                    buffer_length=1,
                    offset=i,
                    ddr_id=ddr_id,
                )
                ipu_write32(S2MM, channel_index, col, bd_id)
                ipu_sync(
                    channel=0,
                    column=0,
                    column_num=1,
                    direction=0,
                    row=0,
                    row_num=1,
                )

        @memtile_dma(tile_0_1)
        def memtile_dma_0_1():
            # input flow
            buffer_0_1_a = aie.buffer(T.memref(k, T.i32()), tile_0_1)
            buffer_0_1_b = aie.buffer(T.memref(k, T.i32()), tile_0_1)
            # output flow
            buffer_0_1_c = aie.buffer(T.memref(1, T.i32()), tile_0_1)

            @dma(S2MM, 0)
            def dma1():
                aie.use_lock(lock_0_1_read_in_a, AcquireGreaterEqual)
                aie.dma_bd(buffer_0_1_a)
                aie.use_lock(lock_0_1_write_out_a, Release)

            @dma(MM2S, 0)
            def dma2():
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

            # output flow
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
            output = memref_dialect.alloc(MemRefType.get([], T.i32()), [], [])
            for _ in range_(0, tiles):
                # wait on both in and out to be ready
                # these have to be acge for some reason...
                aie.use_lock(lock_0_2_use_a, AcquireGreaterEqual)
                aie.use_lock(lock_0_2_use_b, AcquireGreaterEqual)
                aie.use_lock(lock_0_2_use_c, AcquireGreaterEqual)

                fill(arith.constant(0), outs=[output])
                linalg.dot(buffer_0_2_a, buffer_0_2_b, output)
                v = memref_dialect.load(output, [])
                fill(arith.constant(0), outs=[output])
                buffer_0_2_c[0] = v

                aie.use_lock(lock_0_2_read_in_a, Release)
                aie.use_lock(lock_0_2_read_in_b, Release)
                aie.use_lock(lock_0_2_write_out_c, Release)
                yield_([])

    module = run_pipeline(module, Pipeline().canonicalize())
    lowered_linalg = run_pipeline(
        module, Pipeline().convert_linalg_to_loops().fold_memref_alias_ops()
    )
    input_with_addresses = run_pipeline(lowered_linalg, INPUT_WITH_ADDRESSES_PIPELINE)
    print(input_with_addresses)
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
        inps, outps = xclbin.mmap_buffers([(K,), (K,)], [(tiles,)], np.int32)

        wrap_A = np.asarray(inps[0])
        wrap_B = np.asarray(inps[1])
        wrap_C = np.asarray(outps[0])

        A = np.random.randint(0, 10, (K,), dtype=np.int32)
        # B = np.ones((K), dtype=np.int32)
        # A[: K // 2], A[-K // 2 :] = 1, 2
        B = np.random.randint(0, 10, (K,), dtype=np.int32)
        C = np.zeros((tiles,), dtype=np.int32)

        np.copyto(wrap_A, A, casting="no")
        np.copyto(wrap_B, B, casting="no")
        np.copyto(wrap_C, C, casting="no")

        xclbin.sync_buffers_to_device()
        xclbin.run()
        print("Running kernel")
        xclbin.wait(30)
        xclbin.sync_buffers_from_device()

        if not np.array_equal(A @ B, sum(wrap_C)):
            with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
                print(A @ B)
                print(wrap_C)


# CHECK-LABEL: vec_dot_sugar
@construct_and_print_module
def vec_dot_sugar(module):
    K = 32
    tiles = 4
    k = K // tiles

    @device(AIEDevice.ipu)
    def ipu():
        tile_0_0 = tile(0, 0)
        tile_0_1 = tile(0, 1)
        tile_0_2 = tile(0, 2)

        # in
        buffer_0_2_a = aie.buffer(T.memref(k, T.i32()), tile_0_2)
        buffer_0_2_b = aie.buffer(T.memref(k, T.i32()), tile_0_2)
        # out
        buffer_0_2_c = aie.buffer(T.memref(1, T.i32()), tile_0_2)

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
            offsets = list(range(0, K, k))
            for i, bd_id in enumerate(range(tiles)):
                ipu_writebd_shimtile(
                    bd_id,
                    buffer_length=k,
                    offset=offsets[i],
                    ddr_id=ddr_id,
                )
                ipu_write32(MM2S, channel_index, col, bd_id)

            # in B
            channel_index = 1
            col = 0
            ddr_id = 1
            for i, bd_id in enumerate(range(bd_id + 1, bd_id + 1 + tiles)):
                ipu_writebd_shimtile(
                    bd_id,
                    buffer_length=k,
                    offset=offsets[i],
                    ddr_id=ddr_id,
                )
                ipu_write32(MM2S, channel_index, col, bd_id)

            # out C
            channel_index = 0
            col = 0
            ddr_id = 2
            for i, bd_id in enumerate(range(bd_id + 1, bd_id + 1 + tiles)):
                ipu_writebd_shimtile(
                    bd_id,
                    buffer_length=1,
                    offset=i,
                    ddr_id=ddr_id,
                )
                ipu_write32(S2MM, channel_index, col, bd_id)
                ipu_sync(
                    channel=0,
                    column=0,
                    column_num=1,
                    direction=0,
                    row=0,
                    row_num=1,
                )

        @memtile_dma(tile_0_1)
        def memtile_dma_0_1():
            # input flow
            buffer_0_1_a = aie.buffer(T.memref(k, T.i32()), tile_0_1)
            buffer_0_1_b = aie.buffer(T.memref(k, T.i32()), tile_0_1)
            # output flow
            buffer_0_1_c = aie.buffer(T.memref(1, T.i32()), tile_0_1)

            forward_bd(tile_0_1, 0, buffer_0_1_a)
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
            output = memref_dialect.alloc(MemRefType.get([], T.i32()), [], [])
            for _ in range_(0, tiles):
                with (
                    hold_lock(lock_0_2_use_a, lock_0_2_read_in_a),
                    hold_lock(lock_0_2_use_b, lock_0_2_read_in_b),
                    hold_lock(
                        lock_0_2_use_c,
                        lock_0_2_write_out_c,
                    ),
                ):
                    fill(arith.constant(0), outs=[output])
                    linalg.dot(buffer_0_2_a, buffer_0_2_b, output)
                    v = memref_dialect.load(output, [])
                    fill(arith.constant(0), outs=[output])
                    buffer_0_2_c[0] = v

                yield_([])

    module = run_pipeline(module, Pipeline().canonicalize())
    lowered_linalg = run_pipeline(
        module, Pipeline().convert_linalg_to_loops().fold_memref_alias_ops()
    )
    input_with_addresses = run_pipeline(lowered_linalg, INPUT_WITH_ADDRESSES_PIPELINE)
    print(input_with_addresses)
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
        inps, outps = xclbin.mmap_buffers([(K,), (K,)], [(tiles,)], np.int32)

        wrap_A = np.asarray(inps[0])
        wrap_B = np.asarray(inps[1])
        wrap_C = np.asarray(outps[0])

        A = np.random.randint(0, 10, (K,), dtype=np.int32)
        # B = np.ones((K), dtype=np.int32)
        # A[: K // 2], A[-K // 2 :] = 1, 2
        B = np.random.randint(0, 10, (K,), dtype=np.int32)
        C = np.zeros((tiles,), dtype=np.int32)

        np.copyto(wrap_A, A, casting="no")
        np.copyto(wrap_B, B, casting="no")
        np.copyto(wrap_C, C, casting="no")

        xclbin.sync_buffers_to_device()
        xclbin.run()
        print("Running kernel")
        xclbin.wait(30)
        xclbin.sync_buffers_from_device()

        if not np.array_equal(A @ B, sum(wrap_C)):
            with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
                print(A @ B)
                print(wrap_C)
