# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.
# RUN: export BASENAME="$(basename %s .py)"
# RUN: rm -rf "$BASENAME" && mkdir "$BASENAME" && cd "$BASENAME"
# RUN: VITIS_DIR="$VITIS" WORKDIR="$PWD" XRT_DIR="%XRT_DIR" %PYTHON %s


import random
import sys

from aie.extras.util import find_ops

from aie.compiler.aiecc.main import emit_design_kernel_json
from aie.dialects import aie, aiex
from aie.dialects.aie import AIEDevice, DMAChannelDir, LockAction, WireBundle
from aie.dialects.aiex import TileArray
from aie.dialects.linalg.opdsl.ops.core_named_ops import fill as linalg_fill
from aie.dialects.scf import for_ as range_, yield_
from aie.extras.dialects.ext import arith, func, linalg, memref
import aie.extras.types as T
from aie.xrt import XCLBin
from filelock import FileLock
import numpy as np

from util import (
    compile_without_vectorization,
    construct_and_print_module,
    display_flows,
    grouper,
    make_xclbin,
)

DMA = WireBundle.DMA
S2MM = DMAChannelDir.S2MM
MM2S = DMAChannelDir.MM2S
Acquire = LockAction.Acquire
AcquireGreaterEqual = LockAction.AcquireGreaterEqual
Release = LockAction.Release


# CHECK-LABEL: systolic_vec_add
@construct_and_print_module
def systolic_vec_add(module):
    K = 32
    tiles = 1
    k = K // tiles
    columns = [1, 2, 3]
    ipu_insts = aiex.ipu.get_prolog()

    @aie.device(AIEDevice.ipu)
    def ipu():
        if 0 not in columns:
            _dummy_tile = aie.tile(0, 2)

        for column in columns:
            shim_tile = aie.tile(column, 0)
            mem_tile = aie.tile(column, 1)
            compute_tile = aie.tile(column, 2)

            input_a_tile_row_0_to_tile_row_1 = aie.flow(
                shim_tile, DMA, 0, mem_tile, DMA, 0
            )
            input_a_tile_row_1_to_tile_row_2 = aie.flow(
                mem_tile, DMA, 0, compute_tile, DMA, 0
            )
            input_b_tile_row_0_to_tile_row_1 = aie.flow(
                shim_tile, DMA, 1, mem_tile, DMA, 1
            )
            input_b_tile_row_1_to_tile_row_2 = aie.flow(
                mem_tile, DMA, 1, compute_tile, DMA, 1
            )
            # output flow
            output_c_tile_row_2_to_tile_row_1 = aie.flow(
                compute_tile, DMA, 0, mem_tile, DMA, 2
            )
            output_c_tile_row_1_to_tile_row_0 = aie.flow(
                mem_tile, DMA, 2, shim_tile, DMA, 0
            )

            @aie.memtile_dma(mem_tile)
            def memtile_dma_0_1():
                # input flow
                buffer_row_1_a = aie.buffer(
                    T.memref(k, T.i32()), mem_tile, sym_name=f"buffer_{column}_1_a"
                )
                buffer_row_1_b = aie.buffer(
                    T.memref(k, T.i32()), mem_tile, sym_name=f"buffer_{column}_1_b"
                )
                # output flow
                buffer_row_1_c = aie.buffer(
                    T.memref(k, T.i32()), mem_tile, sym_name=f"buffer_{column}_1_c"
                )

                aiex.forward_bd(
                    mem_tile,
                    buffer_row_1_a,
                    input_a_tile_row_0_to_tile_row_1.dest_channel,
                )
                aiex.forward_bd(
                    mem_tile,
                    buffer_row_1_b,
                    input_b_tile_row_0_to_tile_row_1.dest_channel,
                )
                aiex.forward_bd(
                    mem_tile,
                    buffer_row_1_c,
                    output_c_tile_row_1_to_tile_row_0.source_channel,
                )

                aie.end()

            # in
            buffer_row_2_a = aie.buffer(
                T.memref(k, T.i32()), compute_tile, sym_name=f"buffer_{column}_2_a"
            )
            buffer_row_2_b = aie.buffer(
                T.memref(k, T.i32()), compute_tile, sym_name=f"buffer_{column}_2_b"
            )
            # out
            buffer_row_2_c = aie.buffer(
                T.memref(k, T.i32()), compute_tile, sym_name=f"buffer_{column}_2_c"
            )

            lock_row_2_read_in_a = aie.lock(
                compute_tile, lock_id=0, init=1, sym_name=f"lock_{column}_2_read_in_a"
            )
            lock_row_2_use_a = aie.lock(
                compute_tile, lock_id=1, init=0, sym_name=f"lock_{column}_2_use_a"
            )
            lock_row_2_read_in_b = aie.lock(
                compute_tile, lock_id=2, init=1, sym_name=f"lock_{column}_2_read_in_b"
            )
            lock_row_2_use_b = aie.lock(
                compute_tile, lock_id=3, init=0, sym_name=f"lock_{column}_2_use_b"
            )
            lock_row_2_use_c = aie.lock(
                compute_tile, lock_id=4, init=1, sym_name=f"lock_{column}_2_use_c"
            )
            lock_row_2_write_out_c = aie.lock(
                compute_tile, lock_id=5, init=0, sym_name=f"lock_{column}_2_write_out_c"
            )

            @aie.mem(compute_tile)
            def mem_row_2():
                # input
                @aie.dma(S2MM, input_a_tile_row_1_to_tile_row_2.dest_channel)
                def dma1():
                    aiex.process_bd(
                        lock_row_2_read_in_a, buffer_row_2_a, lock_row_2_use_a
                    )

                @aie.dma(S2MM, input_b_tile_row_1_to_tile_row_2.dest_channel)
                def dma2():
                    aiex.process_bd(
                        lock_row_2_read_in_b, buffer_row_2_b, lock_row_2_use_b
                    )

                # output
                @aie.dma(MM2S, output_c_tile_row_2_to_tile_row_1.source_channel)
                def dma3():
                    aiex.process_bd(
                        lock_row_2_write_out_c, buffer_row_2_c, lock_row_2_use_c
                    )

                aie.end()

            @aie.core(compute_tile)
            def core():
                for _ in range_(0, tiles):
                    with (
                        aiex.hold_lock(lock_row_2_use_a, lock_row_2_read_in_a),
                        aiex.hold_lock(lock_row_2_use_b, lock_row_2_read_in_b),
                        aiex.hold_lock(lock_row_2_use_c, lock_row_2_write_out_c),
                    ):
                        linalg_fill(arith.constant(0), outs=[buffer_row_2_c])
                        linalg.add(buffer_row_2_a, buffer_row_2_b, buffer_row_2_c)

                    yield_([])

        # in A
        ddr_id = 0
        for column in columns:
            offsets = list(range(0, K, k))
            for i, bd_id in enumerate(range(tiles)):
                ipu_insts.extend(
                    aiex.ipu.writebd_shimtile(
                        bd_id=bd_id,
                        column=column,
                        buffer_length=k,
                        buffer_offset=offsets[i],
                        ddr_id=ddr_id,
                    )
                )
                ipu_insts.extend(
                    aiex.ipu.shimtile_push_queue(
                        channel_dir=MM2S,
                        channel_index=input_a_tile_row_0_to_tile_row_1.source_channel,
                        column=column,
                        bd_id=bd_id,
                    )
                )
            ddr_id += 1

            # in B
            for i, bd_id in enumerate(range(bd_id + 1, bd_id + 1 + tiles)):
                ipu_insts.extend(
                    aiex.ipu.writebd_shimtile(
                        bd_id=bd_id,
                        column=column,
                        buffer_length=k,
                        buffer_offset=offsets[i],
                        ddr_id=ddr_id,
                    )
                )
                ipu_insts.extend(
                    aiex.ipu.shimtile_push_queue(
                        channel_dir=MM2S,
                        channel_index=input_b_tile_row_0_to_tile_row_1.source_channel,
                        column=column,
                        bd_id=bd_id,
                    )
                )
            ddr_id += 1

            # out C
            for i, bd_id in enumerate(range(bd_id + 1, bd_id + 1 + tiles)):
                ipu_insts.extend(
                    aiex.ipu.writebd_shimtile(
                        bd_id=bd_id,
                        column=column,
                        buffer_length=k,
                        buffer_offset=offsets[i],
                        ddr_id=ddr_id,
                    )
                )
                ipu_insts.extend(
                    aiex.ipu.shimtile_push_queue(
                        channel_dir=S2MM,
                        channel_index=output_c_tile_row_1_to_tile_row_0.dest_channel,
                        column=column,
                        bd_id=bd_id,
                    )
                )
                ipu_insts.extend(
                    aiex.ipu.sync(
                        channel=output_c_tile_row_1_to_tile_row_0.dest_channel,
                        column=column,
                        column_num=1,
                        direction=0,
                        row=0,
                        row_num=1,
                    )
                )
            ddr_id += 1

    compile_without_vectorization(module)
    buffer_args = []
    for c in columns:
        buffer_args.extend([f"a{c}", f"b{c}", f"c{c}"])

    kernel_json = emit_design_kernel_json(buffer_args=buffer_args)
    xclbin_path = make_xclbin(module, kernel_json=kernel_json)
    with FileLock("/tmp/ipu.lock"):
        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        xclbin.load_ipu_instructions(ipu_insts)
        views = xclbin.mmap_buffers([(K,)] * 3 * len(columns), np.int32)

        for a, b, c in grouper(views, 3):
            wrap_A = np.asarray(a)
            wrap_B = np.asarray(b)
            wrap_C = np.asarray(c)

            A = np.random.randint(0, 10, (K,), dtype=np.int32)
            B = np.random.randint(0, 10, (K,), dtype=np.int32)
            C = np.zeros((K,), dtype=np.int32)

            np.copyto(wrap_A, A, casting="no")
            np.copyto(wrap_B, B, casting="no")
            np.copyto(wrap_C, C, casting="no")

        xclbin.sync_buffers_to_device()
        xclbin.run()
        print("Running kernel")
        xclbin.wait(30)
        xclbin.sync_buffers_from_device()

    with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
        for a, b, c in grouper(views, 3):
            wrap_A = np.asarray(a)
            wrap_B = np.asarray(b)
            wrap_C = np.asarray(c)
            print(f"a={wrap_A}")
            print(f"b={wrap_B}")
            print(f"c={wrap_C}")


# CHECK-LABEL: max_result_args
@construct_and_print_module
def max_result_args(module):
    K = 32
    tiles = 1
    k = K // tiles
    columns = [0, 1, 2, 3]
    RANDOM_NUMBER = random.randint(1, 100)
    print(f"{RANDOM_NUMBER=}")
    ipu_insts = aiex.ipu.get_prolog()

    @aie.device(AIEDevice.ipu)
    def ipu():
        if 0 not in columns:
            _dummy_tile = aie.tile(0, 2)

        for column in columns:
            shim_tile = aie.tile(column, 0)
            mem_tile = aie.tile(column, 1)
            compute_tile = aie.tile(column, 2)

            # output flow
            output_c_tile_row_2_to_tile_row_1 = aie.flow(
                compute_tile, DMA, 0, mem_tile, DMA, 2
            )
            output_c_tile_row_1_to_tile_row_0 = aie.flow(
                mem_tile, DMA, 2, shim_tile, DMA, 0
            )

            @aie.memtile_dma(mem_tile)
            def memtile_dma_0_1():
                # output flow
                buffer_row_1_c = aie.buffer(
                    T.memref(k, T.i32()), mem_tile, sym_name=f"buffer_{column}_1_c"
                )

                aiex.forward_bd(
                    mem_tile,
                    buffer_row_1_c,
                    output_c_tile_row_1_to_tile_row_0.source_channel,
                )

                aie.end()

            # out
            buffer_row_2_c = aie.buffer(
                T.memref(k, T.i32()), compute_tile, sym_name=f"buffer_{column}_2_c"
            )
            lock_row_2_use_c = aie.lock(
                compute_tile, lock_id=4, init=1, sym_name=f"lock_{column}_2_use_c"
            )
            lock_row_2_write_out_c = aie.lock(
                compute_tile, lock_id=5, init=0, sym_name=f"lock_{column}_2_write_out_c"
            )

            @aie.mem(compute_tile)
            def mem_row_2():
                # output
                @aie.dma(MM2S, output_c_tile_row_2_to_tile_row_1.source_channel)
                def dma3():
                    aiex.process_bd(
                        lock_row_2_write_out_c, buffer_row_2_c, lock_row_2_use_c
                    )

                aie.end()

            @aie.core(compute_tile)
            def core():
                for _ in range_(0, tiles):
                    with aiex.hold_lock(lock_row_2_use_c, lock_row_2_write_out_c):
                        linalg_fill(
                            arith.constant(column + RANDOM_NUMBER),
                            outs=[buffer_row_2_c],
                        )

                    yield_([])

        ddr_id = 0
        for column in columns:
            offsets = list(range(0, K, k))
            bd_id = 0
            # out C
            for i, bd_id in enumerate(range(bd_id + 1, bd_id + 1 + tiles)):
                ipu_insts.extend(
                    aiex.ipu.writebd_shimtile(
                        bd_id=bd_id,
                        column=column,
                        buffer_length=k,
                        buffer_offset=offsets[i],
                        ddr_id=ddr_id,
                    )
                )
                ipu_insts.extend(
                    aiex.ipu.shimtile_push_queue(
                        channel_dir=S2MM,
                        channel_index=output_c_tile_row_1_to_tile_row_0.dest_channel,
                        column=column,
                        bd_id=bd_id,
                    )
                )
                ipu_insts.extend(
                    aiex.ipu.sync(
                        channel=output_c_tile_row_1_to_tile_row_0.dest_channel,
                        column=column,
                        column_num=1,
                        direction=0,
                        row=0,
                        row_num=1,
                    )
                )
            ddr_id += 1

    compile_without_vectorization(module)
    buffer_args = [f"c{c}" for c in columns]
    kernel_json = emit_design_kernel_json(buffer_args=buffer_args)
    xclbin_path = make_xclbin(module, kernel_json=kernel_json)
    with FileLock("/tmp/ipu.lock"):
        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        xclbin.load_ipu_instructions(ipu_insts)
        views = xclbin.mmap_buffers([(K,)] * len(columns), np.int32)

        for c in views:
            wrap_C = np.asarray(c)
            C = np.zeros((K,), dtype=np.int32)
            np.copyto(wrap_C, C, casting="no")

        xclbin.sync_buffers_to_device()
        xclbin.run()
        print("Running kernel")
        xclbin.wait(30)
        xclbin.sync_buffers_from_device()

    with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
        for c in views:
            wrap_C = np.asarray(c)
            print(f"c={wrap_C}")


# CHECK-LABEL: max_input_and_output_args
@construct_and_print_module
def max_input_and_output_args(module):
    K = 32
    tiles = 1
    k = K // tiles
    columns = [1, 2, 3]
    ipu_insts = aiex.ipu.get_prolog()

    @aie.device(AIEDevice.ipu)
    def ipu():
        if 0 not in columns:
            _dummy_tile = aie.tile(0, 2)

        for column in columns:
            shim_tile = aie.tile(column, 0)
            mem_tile = aie.tile(column, 1)
            compute_tile = aie.tile(column, 2)

            input_a_tile_row_0_to_tile_row_1 = aie.flow(
                shim_tile, DMA, 0, mem_tile, DMA, 0
            )
            input_a_tile_row_1_to_tile_row_2 = aie.flow(
                mem_tile, DMA, 0, compute_tile, DMA, 0
            )
            # output flow
            output_c_tile_row_2_to_tile_row_1 = aie.flow(
                compute_tile, DMA, 0, mem_tile, DMA, 2
            )
            output_c_tile_row_1_to_tile_row_0 = aie.flow(
                mem_tile, DMA, 2, shim_tile, DMA, 0
            )

            @aie.memtile_dma(mem_tile)
            def memtile_dma_0_1():
                # input flow
                buffer_row_1_a = aie.buffer(
                    T.memref(k, T.i32()), mem_tile, sym_name=f"buffer_{column}_1_a"
                )
                # output flow
                buffer_row_1_c = aie.buffer(
                    T.memref(k, T.i32()), mem_tile, sym_name=f"buffer_{column}_1_c"
                )

                aiex.forward_bd(
                    mem_tile,
                    buffer_row_1_a,
                    input_a_tile_row_0_to_tile_row_1.dest_channel,
                )
                aiex.forward_bd(
                    mem_tile,
                    buffer_row_1_c,
                    output_c_tile_row_1_to_tile_row_0.source_channel,
                )

                aie.end()

            # in
            buffer_row_2_a = aie.buffer(
                T.memref(k, T.i32()), compute_tile, sym_name=f"buffer_{column}_2_a"
            )
            # out
            buffer_row_2_c = aie.buffer(
                T.memref(k, T.i32()), compute_tile, sym_name=f"buffer_{column}_2_c"
            )

            lock_row_2_read_in_a = aie.lock(
                compute_tile, lock_id=0, init=1, sym_name=f"lock_{column}_2_read_in_a"
            )
            lock_row_2_use_a = aie.lock(
                compute_tile, lock_id=1, init=0, sym_name=f"lock_{column}_2_use_a"
            )
            lock_row_2_use_c = aie.lock(
                compute_tile, lock_id=4, init=1, sym_name=f"lock_{column}_2_use_c"
            )
            lock_row_2_write_out_c = aie.lock(
                compute_tile, lock_id=5, init=0, sym_name=f"lock_{column}_2_write_out_c"
            )

            @aie.mem(compute_tile)
            def mem_row_2():
                # input
                @aie.dma(S2MM, input_a_tile_row_1_to_tile_row_2.dest_channel)
                def dma1():
                    aiex.process_bd(
                        lock_row_2_read_in_a, buffer_row_2_a, lock_row_2_use_a
                    )

                # output
                @aie.dma(MM2S, output_c_tile_row_2_to_tile_row_1.source_channel)
                def dma3():
                    aiex.process_bd(
                        lock_row_2_write_out_c, buffer_row_2_c, lock_row_2_use_c
                    )

                aie.end()

            @aie.core(compute_tile)
            def core():
                for _ in range_(0, tiles):
                    with (
                        aiex.hold_lock(lock_row_2_use_a, lock_row_2_read_in_a),
                        aiex.hold_lock(lock_row_2_use_c, lock_row_2_write_out_c),
                    ):
                        linalg_fill(arith.constant(0), outs=[buffer_row_2_c])
                        linalg.copy(buffer_row_2_a, buffer_row_2_c)

                    yield_([])

        # in A
        ddr_id = 0
        for column in columns:
            offsets = list(range(0, K, k))
            for i, bd_id in enumerate(range(tiles)):
                ipu_insts.extend(
                    aiex.ipu.writebd_shimtile(
                        bd_id=bd_id,
                        column=column,
                        buffer_length=k,
                        buffer_offset=offsets[i],
                        ddr_id=ddr_id,
                    )
                )
                ipu_insts.extend(
                    aiex.ipu.shimtile_push_queue(
                        channel_dir=MM2S,
                        channel_index=input_a_tile_row_0_to_tile_row_1.source_channel,
                        column=column,
                        bd_id=bd_id,
                    )
                )
            ddr_id += 1

            # out C
            for i, bd_id in enumerate(range(bd_id + 1, bd_id + 1 + tiles)):
                ipu_insts.extend(
                    aiex.ipu.writebd_shimtile(
                        bd_id=bd_id,
                        column=column,
                        buffer_length=k,
                        buffer_offset=offsets[i],
                        ddr_id=ddr_id,
                    )
                )
                ipu_insts.extend(
                    aiex.ipu.shimtile_push_queue(
                        channel_dir=S2MM,
                        channel_index=output_c_tile_row_1_to_tile_row_0.dest_channel,
                        column=column,
                        bd_id=bd_id,
                    )
                )
                ipu_insts.extend(
                    aiex.ipu.sync(
                        channel=output_c_tile_row_1_to_tile_row_0.dest_channel,
                        column=column,
                        column_num=1,
                        direction=0,
                        row=0,
                        row_num=1,
                    )
                )
            ddr_id += 1

    compile_without_vectorization(module)
    buffer_args = []
    for c in columns:
        buffer_args.extend([f"a{c}", f"c{c}"])

    kernel_json = emit_design_kernel_json(buffer_args=buffer_args)
    xclbin_path = make_xclbin(module, kernel_json=kernel_json)
    with FileLock("/tmp/ipu.lock"):
        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        xclbin.load_ipu_instructions(ipu_insts)
        views = xclbin.mmap_buffers([(K,)] * 2 * len(columns), np.int32)

        for i, (a, c) in enumerate(grouper(views, 2), start=1):
            wrap_A = np.asarray(a)
            wrap_C = np.asarray(c)

            # A = np.random.randint(0, 10, (K,), dtype=np.int32)
            A = np.ones((K,), dtype=np.int32) * i
            C = np.zeros((K,), dtype=np.int32)

            np.copyto(wrap_A, A, casting="no")
            np.copyto(wrap_C, C, casting="no")

        xclbin.sync_buffers_to_device()
        xclbin.run()
        print("Running kernel")
        xclbin.wait(30)
        xclbin.sync_buffers_from_device()

    with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
        for a, c in grouper(views, 2):
            wrap_A = np.asarray(a)
            wrap_C = np.asarray(c)
            print(f"a={wrap_A}")
            print(f"c={wrap_C}")


# CHECK-LABEL: zeroth_column
@construct_and_print_module
def zeroth_column(module):
    RANDOM_NUMBER = random.randint(1, 100)
    print(RANDOM_NUMBER)
    K = 32
    ipu_insts = aiex.ipu.get_prolog()

    @aie.device(AIEDevice.ipu)
    def ipu():
        shim_tile = aie.tile(1, 0)
        mem_tile = aie.tile(0, 1)
        compute_tile = aie.tile(0, 2)

        # output flow
        output_c_tile_row_2_to_tile_row_1 = aie.flow(
            compute_tile, DMA, 0, mem_tile, DMA, 2
        )
        output_c_tile_row_1_to_tile_row_0 = aie.flow(
            mem_tile, DMA, 2, shim_tile, DMA, 0
        )

        @aie.memtile_dma(mem_tile)
        def memtile_dma_0_1():
            buffer_row_1_c = aie.buffer(T.memref(K, T.i32()), mem_tile)
            aiex.forward_bd(
                mem_tile,
                buffer_row_1_c,
                output_c_tile_row_1_to_tile_row_0.source_channel,
            )
            aie.end()

        # out
        buffer_row_2_c = aie.buffer(T.memref(K, T.i32()), compute_tile)
        lock_row_2_use_c = aie.lock(compute_tile, lock_id=4, init=1)
        lock_row_2_write_out_c = aie.lock(compute_tile, lock_id=5, init=0)

        @aie.mem(compute_tile)
        def mem_row_2():
            # output
            @aie.dma(MM2S, output_c_tile_row_2_to_tile_row_1.source_channel)
            def dma3():
                aiex.process_bd(
                    lock_row_2_write_out_c, buffer_row_2_c, lock_row_2_use_c
                )

            aie.end()

        @aie.core(compute_tile)
        def core():
            with aiex.hold_lock(lock_row_2_use_c, lock_row_2_write_out_c):
                linalg_fill(
                    arith.constant(RANDOM_NUMBER),
                    outs=[buffer_row_2_c],
                )

        ddr_id = 0
        offset = 0
        bd_id = 0
        # out C
        ipu_insts.extend(
            aiex.ipu.writebd_shimtile(
                bd_id=bd_id,
                column=int(output_c_tile_row_1_to_tile_row_0.dest.owner.opview.col),
                buffer_length=K,
                buffer_offset=offset,
                ddr_id=ddr_id,
            )
        )
        ipu_insts.extend(
            aiex.ipu.shimtile_push_queue(
                channel_dir=S2MM,
                channel_index=output_c_tile_row_1_to_tile_row_0.dest_channel,
                column=int(output_c_tile_row_1_to_tile_row_0.dest.owner.opview.col),
                bd_id=bd_id,
            )
        )
        ipu_insts.extend(
            aiex.ipu.sync(
                channel=output_c_tile_row_1_to_tile_row_0.dest_channel,
                column=int(output_c_tile_row_1_to_tile_row_0.dest.owner.opview.col),
                column_num=1,
                direction=0,
                row=0,
                row_num=1,
            )
        )
        ddr_id += 1

    compile_without_vectorization(module, partition_start_col=0)
    buffer_args = ["output"]
    kernel_json = emit_design_kernel_json(buffer_args=buffer_args)
    xclbin_path = make_xclbin(module, kernel_json=kernel_json, start_columns=[0])
    with FileLock("/tmp/ipu.lock"):
        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        xclbin.load_ipu_instructions(ipu_insts)
        views = xclbin.mmap_buffers([(K,)], np.int32)
        assert len(views) == 1
        view = views[0]

        wrap_view = np.asarray(view)
        C = np.zeros((K,), dtype=np.int32)
        np.copyto(wrap_view, C, casting="no")

        xclbin.sync_buffers_to_device()
        xclbin.run()
        print("Running kernel")
        xclbin.wait(30)
        xclbin.sync_buffers_from_device()

    with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
        print(f"{wrap_view=}")


# CHECK-LABEL: global_core_mem_init
@construct_and_print_module
def global_core_mem_init(module):
    K = 32
    tiles = 1
    k = K // tiles
    columns = [0, 1, 2, 3]
    RANDOM_NUMBER = random.randint(1, 100)
    print(RANDOM_NUMBER)
    ipu_insts = aiex.ipu.get_prolog()

    @aie.device(AIEDevice.ipu)
    def ipu():
        if 0 not in columns:
            _dummy_tile = aie.tile(0, 2)

        for column in columns:
            shim_tile = aie.tile(column, 0)
            mem_tile = aie.tile(column, 1)
            compute_tile = aie.tile(column, 2)

            weight = memref.global_(
                sym_name=f"weight_{column}",
                initial_value=np.ones((k,), dtype=np.int32) * column * RANDOM_NUMBER,
                constant=True,
            ).opview

            # output flow
            output_c_tile_row_2_to_tile_row_1 = aie.flow(
                compute_tile, DMA, 0, mem_tile, DMA, 2
            )
            output_c_tile_row_1_to_tile_row_0 = aie.flow(
                mem_tile, DMA, 2, shim_tile, DMA, 0
            )

            @aie.memtile_dma(mem_tile)
            def memtile_dma_0_1():
                # output flow
                buffer_row_1_c = aie.buffer(
                    T.memref(k, T.i32()), mem_tile, sym_name=f"buffer_{column}_1_c"
                )

                aiex.forward_bd(
                    mem_tile,
                    buffer_row_1_c,
                    output_c_tile_row_1_to_tile_row_0.source_channel,
                )

                aie.end()

            # out
            buffer_row_2_c = aie.buffer(
                T.memref(k, T.i32()), compute_tile, sym_name=f"buffer_{column}_2_c"
            )
            lock_row_2_use_c = aie.lock(
                compute_tile, lock_id=4, init=1, sym_name=f"lock_{column}_2_use_c"
            )
            lock_row_2_write_out_c = aie.lock(
                compute_tile, lock_id=5, init=0, sym_name=f"lock_{column}_2_write_out_c"
            )

            @aie.mem(compute_tile)
            def mem_row_2():
                # output
                @aie.dma(MM2S, output_c_tile_row_2_to_tile_row_1.source_channel)
                def dma3():
                    aiex.process_bd(
                        lock_row_2_write_out_c, buffer_row_2_c, lock_row_2_use_c
                    )

                aie.end()

            @aie.core(compute_tile)
            def core():
                for _ in range_(0, tiles):
                    with aiex.hold_lock(lock_row_2_use_c, lock_row_2_write_out_c):
                        w = memref.get_global(weight.type_.value, weight.sym_name.value)
                        linalg.copy(w, buffer_row_2_c)

                    yield_([])

        ddr_id = 0
        for column in columns:
            offsets = list(range(0, K, k))
            bd_id = 0
            # out C
            for i, bd_id in enumerate(range(bd_id + 1, bd_id + 1 + tiles)):
                ipu_insts.extend(
                    aiex.ipu.writebd_shimtile(
                        bd_id=bd_id,
                        column=column,
                        buffer_length=k,
                        buffer_offset=offsets[i],
                        ddr_id=ddr_id,
                    )
                )
                ipu_insts.extend(
                    aiex.ipu.shimtile_push_queue(
                        channel_dir=S2MM,
                        channel_index=output_c_tile_row_1_to_tile_row_0.dest_channel,
                        column=column,
                        bd_id=bd_id,
                    )
                )
                ipu_insts.extend(
                    aiex.ipu.sync(
                        channel=output_c_tile_row_1_to_tile_row_0.dest_channel,
                        column=column,
                        column_num=1,
                        direction=0,
                        row=0,
                        row_num=1,
                    )
                )
            ddr_id += 1

    compile_without_vectorization(module)
    buffer_args = [f"c{c}" for c in columns]
    kernel_json = emit_design_kernel_json(buffer_args=buffer_args)
    xclbin_path = make_xclbin(module, kernel_json=kernel_json)
    with FileLock("/tmp/ipu.lock"):
        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        xclbin.load_ipu_instructions(ipu_insts)
        views = xclbin.mmap_buffers([(K,)] * len(columns), np.int32)

        for c in views:
            wrap_C = np.asarray(c)
            C = np.zeros((K,), dtype=np.int32)
            np.copyto(wrap_C, C, casting="no")

        xclbin.sync_buffers_to_device()
        xclbin.run()
        print("Running kernel")
        xclbin.wait(30)
        xclbin.sync_buffers_from_device()

    with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
        for c in views:
            wrap_C = np.asarray(c)
            print(f"c={wrap_C}")
