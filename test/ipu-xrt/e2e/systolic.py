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
                    aiex.ipu.write32(
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
                    aiex.ipu.write32(
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
                    aiex.ipu.write32(
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
                    aiex.ipu.write32(
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
                    aiex.ipu.write32(
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
                    aiex.ipu.write32(
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
            aiex.ipu.write32(
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
                    aiex.ipu.write32(
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


# CHECK-LABEL: constant_systolic_3x3
@construct_and_print_module
def constant_systolic_3x3(module):
    K = 32
    n_tiles = 1
    columns = [0, 1, 2]
    rows = list(range(5))
    # six rows X five columns
    # image orientation x-> cols, y-> rows
    tiles = np.empty((5, 6), dtype=object)
    tiles[0, 0] = None
    repeat_count = 2
    ipu_insts = aiex.ipu.get_prolog()

    @aie.device(AIEDevice.ipu)
    def ipu():
        weight_id = random.randint(1, 100)
        for c in columns:
            for r in rows:
                if (c, r) == (0, 0):
                    continue
                tiles[c, r] = aie.tile(c, r)

        for r in [2, 3]:
            for c in columns[:-1]:
                tiles[c, r] >> tiles[c + 1, r]

        for c in columns[1:]:
            for r in rows[:-1]:
                tiles[c, r] << tiles[c, r + 1]

        for t in tiles.flat:
            if t is None or t.flows is None:
                continue
            for _, v in t.flows.items():
                assert len(v) == 1, print(v)

        for r in [2, 3]:
            compute_tile = tiles[0, r]
            next_tile_next_col = tiles[0 + 1, r]
            weight = memref.global_(
                sym_name=f"weight_0_{r}",
                initial_value=np.ones((K,), dtype=np.int32) * weight_id,
                constant=True,
            )
            weight_id = random.randint(1, 10)
            buffer_weight = aie.buffer(
                T.memref(K, T.i32()), compute_tile, sym_name=f"buffer_0_{r}_weight"
            )
            lock_read_weight = aie.lock(
                compute_tile, init=1, sym_name=f"lock_0_{r}_read_weight"
            )
            lock_send_weight = aie.lock(
                compute_tile, init=0, sym_name=f"lock_0_{r}_send_weight"
            )

            @aie.mem(compute_tile)
            def mem():
                @aie.dma(
                    MM2S,
                    compute_tile.flows[next_tile_next_col][0].source_channel,
                    loop=False,
                    repeat_count=repeat_count,
                )
                def _():
                    aiex.process_bd(lock_send_weight, buffer_weight, lock_read_weight)

                aie.end()

            @aie.core(compute_tile)
            def core():
                for _ in range(repeat_count):
                    with aiex.hold_lock(lock_read_weight, lock_send_weight):
                        x = memref.get_global(weight.type_.value, weight.sym_name.value)
                        linalg.copy(x, buffer_weight)

        for c in [1, 2]:
            compute_tile = tiles[c, 4]
            next_tile_next_row = tiles[c, 4 - 1]
            weight = memref.global_(
                sym_name=f"weight_{c}_4",
                initial_value=np.ones((K,), dtype=np.int32) * weight_id,
                constant=True,
            )
            weight_id = random.randint(1, 10)
            buffer_weight = aie.buffer(
                T.memref(K, T.i32()), compute_tile, sym_name=f"buffer_{c}_4_weight"
            )
            lock_read_weight = aie.lock(
                compute_tile, init=1, sym_name=f"lock_{c}_4_read_weight"
            )
            lock_send_weight = aie.lock(
                compute_tile, init=0, sym_name=f"lock_{c}_4_send_weight"
            )

            @aie.mem(compute_tile)
            def mem():
                @aie.dma(
                    MM2S,
                    compute_tile.flows[next_tile_next_row][0].source_channel,
                    loop=False,
                    repeat_count=repeat_count,
                )
                def _():
                    aiex.process_bd(lock_send_weight, buffer_weight, lock_read_weight)

                aie.end()

            @aie.core(compute_tile)
            def core():
                for _ in range(repeat_count):
                    with aiex.hold_lock(lock_read_weight, lock_send_weight):
                        x = memref.get_global(weight.type_.value, weight.sym_name.value)
                        linalg.copy(x, buffer_weight)

        for c in [1, 2]:
            for r in [2, 3]:
                compute_tile = tiles[c, r]
                prev_tile_prev_col = tiles[c - 1, r]
                next_tile_next_col = tiles[c + 1, r]
                prev_tile_prev_row = tiles[c, r + 1]
                next_tile_next_row = tiles[c, r - 1]

                buffer_a = aie.buffer(
                    T.memref(K, T.i32()), compute_tile, sym_name=f"buffer_{c}_{r}_a"
                )
                buffer_b = aie.buffer(
                    T.memref(K, T.i32()), compute_tile, sym_name=f"buffer_{c}_{r}_b"
                )
                buffer_c = aie.buffer(
                    T.memref(K, T.i32()), compute_tile, sym_name=f"buffer_{c}_{r}_c"
                )

                if next_tile_next_col:
                    lock_read_in_a_init = 2
                else:
                    lock_read_in_a_init = 1
                lock_read_in_a = aie.lock(
                    compute_tile,
                    init=lock_read_in_a_init,
                    sym_name=f"lock_{c}_{r}_read_a",
                )
                lock_use_a = aie.lock(
                    compute_tile, init=0, sym_name=f"lock_{c}_{r}_use_a"
                )

                lock_read_in_b = aie.lock(
                    compute_tile, init=2, sym_name=f"lock_{c}_{r}_read_b"
                )
                lock_use_b = aie.lock(
                    compute_tile, init=0, sym_name=f"lock_{c}_{r}_use_b"
                )

                lock_use_c = aie.lock(
                    compute_tile, init=1, sym_name=f"lock_{c}_{r}_use_c"
                )
                lock_write_out_c = aie.lock(
                    compute_tile, init=0, sym_name=f"lock_{c}_{r}_write_out_c"
                )

                @aie.mem(compute_tile)
                def mem():
                    @aie.dma(
                        S2MM,
                        compute_tile.flows[prev_tile_prev_col][0].dest_channel,
                        loop=False,
                        repeat_count=repeat_count,
                    )
                    def _():
                        aiex.process_bd(
                            lock_read_in_a,
                            buffer_a,
                            lock_use_a,
                            acq_val=lock_read_in_a_init,
                            rel_val=lock_read_in_a_init,
                        )

                    if next_tile_next_col:

                        @aie.dma(
                            MM2S,
                            compute_tile.flows[next_tile_next_col][0].source_channel,
                            loop=False,
                            repeat_count=repeat_count,
                        )
                        def _():
                            aiex.process_bd(
                                lock_use_a,
                                buffer_a,
                                lock_read_in_a,
                                acq_val=1,
                                rel_val=1,
                            )

                    @aie.dma(
                        S2MM,
                        compute_tile.flows[prev_tile_prev_row][0].dest_channel,
                        loop=False,
                        repeat_count=repeat_count,
                    )
                    def _():
                        aiex.process_bd(
                            lock_read_in_b, buffer_b, lock_use_b, acq_val=2, rel_val=2
                        )

                    @aie.dma(
                        MM2S,
                        compute_tile.flows[next_tile_next_row][0].source_channel,
                        num_blocks=2,
                        loop=False,
                    )
                    def output():
                        aiex.process_bd(
                            lock_use_b,
                            buffer_b,
                            lock_read_in_b,
                            acq_val=1,
                            rel_val=1,
                        )

                    @aie.another_bd(output)
                    def output_c():
                        aiex.process_bd(
                            lock_write_out_c,
                            buffer_c,
                            lock_use_c,
                            acq_val=1,
                            rel_val=1,
                        )

                    aie.end()

                @aie.core(compute_tile)
                def core():
                    with (
                        aiex.hold_lock(
                            lock_use_a, lock_read_in_a, acq_val=1, rel_val=1
                        ),
                        aiex.hold_lock(
                            lock_use_b, lock_read_in_b, acq_val=1, rel_val=1
                        ),
                        aiex.hold_lock(
                            lock_use_c, lock_write_out_c, acq_val=1, rel_val=1
                        ),
                    ):
                        linalg.add(buffer_a, buffer_b, buffer_c)

            compute_tile = tiles[c, 2]
            mem_tile = tiles[c, 1]
            shim_tile = tiles[c, 0]

            @aie.memtile_dma(mem_tile)
            def memtile_dma():
                aiex.forward_bd(
                    mem_tile,
                    aie.buffer(
                        T.memref(K, T.i32()), mem_tile, sym_name=f"buffer_{c}_1_c"
                    ),
                    s2mm_channel_idx=mem_tile.flows[compute_tile][0].dest_channel,
                    mm2s_channel_idx=mem_tile.flows[shim_tile][0].source_channel,
                    repeat_count=repeat_count,
                )

                aie.end()

            arith.constant(80081355)

        for j in range(repeat_count):
            ddr_id = 0
            bd_id = 0
            for column in [1, 2]:
                shim_tile = tiles[c, 0]
                mem_tile = tiles[c, 1]
                # out C
                for i, bd_id in enumerate(range(bd_id + 1, bd_id + 1 + n_tiles)):
                    ipu_insts.extend(
                        aiex.ipu.writebd_shimtile(
                            bd_id=bd_id,
                            column=column,
                            buffer_length=K,
                            buffer_offset=j * K,
                            ddr_id=ddr_id,
                        )
                    )
                    ipu_insts.extend(
                        aiex.ipu.write32(
                            channel_dir=S2MM,
                            channel_index=mem_tile.flows[shim_tile][0].dest_channel,
                            column=column,
                            bd_id=bd_id,
                        )
                    )
                    ipu_insts.extend(
                        aiex.ipu.sync(
                            channel=mem_tile.flows[shim_tile][0].dest_channel,
                            column=column,
                            column_num=1,
                            direction=0,
                            row=0,
                            row_num=1,
                        )
                    )
                ddr_id += 1

    # display_flows(module)
    print(module)
    assert module.operation.verify()

    weights = find_ops(module.operation, lambda o: "memref.global" in str(o))
    for w in weights:
        print(w)

    compile_without_vectorization(module, partition_start_col=0)
    buffer_args = [f"c{c}" for c in [1, 2]]
    kernel_json = emit_design_kernel_json(buffer_args=buffer_args)
    xclbin_path = make_xclbin(module, kernel_json=kernel_json)
    with FileLock("/tmp/ipu.lock"):
        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        xclbin.load_ipu_instructions(ipu_insts)
        views = xclbin.mmap_buffers([(2 * K,)] * len([1, 2]), np.int32)

        for c in views:
            wrap_C = np.asarray(c)
            C = np.zeros((2 * K,), dtype=np.int32)
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
