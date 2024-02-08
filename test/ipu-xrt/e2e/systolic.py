# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.
import sys

# RUN: export BASENAME="$(basename %s .py)"
# RUN: rm -rf "$BASENAME" && mkdir "$BASENAME" && cd "$BASENAME"
# RUN: VITIS_DIR="$VITIS" WORKDIR="$PWD" XRT_DIR="%XRT_DIR" %PYTHON %s


from aie.extras.dialects.ext import arith, func, linalg
from filelock import FileLock
import numpy as np

from aie.compiler.aiecc.main import emit_design_kernel_json
from aie.dialects import aie, aiex
from aie.dialects.aie import AIEDevice, DMAChannelDir, LockAction, WireBundle
from aie.dialects.linalg.opdsl.ops.core_named_ops import fill as linalg_fill
from aie.dialects.scf import for_ as range_, yield_
import aie.extras.types as T
from aie.xrt import XCLBin
from util import (
    compile_without_vectorization,
    construct_and_print_module,
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
                    input_a_tile_row_0_to_tile_row_1.dest_channel,
                    buffer_row_1_a,
                )
                aiex.forward_bd(
                    mem_tile,
                    input_b_tile_row_0_to_tile_row_1.dest_channel,
                    buffer_row_1_b,
                )
                aiex.forward_bd(
                    mem_tile,
                    output_c_tile_row_1_to_tile_row_0.source_channel,
                    buffer_row_1_c,
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

        @func.func(emit=True)
        def bobsyouruncle():
            # in A
            ddr_id = 0
            for column in columns:
                offsets = list(range(0, K, k))
                for i, bd_id in enumerate(range(tiles)):
                    aiex.ipu.writebd_shimtile(
                        bd_id=bd_id,
                        column=column,
                        buffer_length=k,
                        offset=offsets[i],
                        ddr_id=ddr_id,
                    )
                    aiex.ipu.write32(
                        channel_dir=MM2S,
                        channel_index=input_a_tile_row_0_to_tile_row_1.source_channel,
                        column=column,
                        bd_id=bd_id,
                    )
                ddr_id += 1

                # in B
                for i, bd_id in enumerate(range(bd_id + 1, bd_id + 1 + tiles)):
                    aiex.ipu.writebd_shimtile(
                        bd_id=bd_id,
                        column=column,
                        buffer_length=k,
                        offset=offsets[i],
                        ddr_id=ddr_id,
                    )
                    aiex.ipu.write32(
                        channel_dir=MM2S,
                        channel_index=input_b_tile_row_0_to_tile_row_1.source_channel,
                        column=column,
                        bd_id=bd_id,
                    )
                ddr_id += 1

                # out C
                for i, bd_id in enumerate(range(bd_id + 1, bd_id + 1 + tiles)):
                    aiex.ipu.writebd_shimtile(
                        bd_id=bd_id,
                        column=column,
                        buffer_length=k,
                        offset=offsets[i],
                        ddr_id=ddr_id,
                    )
                    aiex.ipu.write32(
                        channel_dir=S2MM,
                        channel_index=output_c_tile_row_1_to_tile_row_0.dest_channel,
                        column=column,
                        bd_id=bd_id,
                    )
                    aiex.ipu.sync(
                        channel=output_c_tile_row_1_to_tile_row_0.dest_channel,
                        column=column,
                        column_num=1,
                        direction=0,
                        row=0,
                        row_num=1,
                    )
                ddr_id += 1

    ipu_insts = compile_without_vectorization(module)
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
