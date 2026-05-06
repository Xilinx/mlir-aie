#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, bneck_13_InC1.

import argparse
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.extras.dialects import memref, arith
from aie.extras.context import mlir_mod_ctx
import math

import aie.utils.trace as trace_utils

from aie2_bottleneckC import bottleneckCCore

sys.path.append("..")
import mb_utils


def create_tile(col, row):
    # Replace this with the appropriate constructor or conversion
    return aie.dialects.aie.tile(col, row)


def select_cores(start_col, start_row):
    # Initialize the list to store the selected cores
    selected_cores = []

    # Current position
    current_col = start_col
    current_row = start_row

    # Direction flag for snake-like pattern
    downward = True

    # Loop to select the next 9 cores
    for _ in range(9):
        # Add the current core to the list
        selected_cores.append((current_col, current_row))

        # Move to the next core based on the direction
        if downward:
            current_row += 1
            if current_row > 5:  # If we reach the bottom boundary
                current_row = 5
                current_col += 1
                downward = False  # Change direction
        else:
            current_row -= 1
            if current_row < 2:  # If we reach the top boundary
                current_row = 2
                current_col += 1
                downward = True  # Change direction

        # If the column index exceeds the limit, break the loop
        if current_col > 7:
            break

    return selected_cores


def aie2_bn_13_14(
    start_row=2,
    start_col=0,
    bn13_scaleFactor1=10,
    bn13_scaleFactor2=7,
    bn13_scaleFactor3=9,
    bn13_scaleFactorAdd=1,
    bn14_scaleFactor1=9,
    bn14_scaleFactor2=8,
    bn14_scaleFactor3=12,
    bn14_scaleFactorAdd=1,
):

    bneck_13_InW1 = 7
    bneck_13_InH1 = 7
    bneck_13_InC1 = 80
    bneck_13_OutC1 = 960
    InputSplit = 2
    OutputSplit = 2  # split output channels based on your preference

    RepeatChannels = math.floor(bneck_13_InH1)

    bneck_13_InW2 = bneck_13_InW1
    bneck_13_InH2 = bneck_13_InH1
    bneck_13_OutC2 = bneck_13_OutC1

    bneck_13_InW3 = bneck_13_InW2
    bneck_13_InH3 = bneck_13_InH2
    bneck_13_OutC3 = 80

    # second block
    bneck_14_InW1 = bneck_13_InW1
    bneck_14_InH1 = bneck_13_InH1
    bneck_14_InC1 = bneck_13_OutC3
    bneck_14_OutC1 = 960

    OutputSplit2 = 2  # split output channels based on your preference

    bneck_14_InW2 = bneck_14_InW1
    bneck_14_InH2 = bneck_14_InH1
    bneck_14_OutC2 = bneck_14_OutC1

    bneck_14_InW3 = bneck_14_InW2
    bneck_14_InH3 = bneck_14_InH2
    bneck_14_OutC3 = 80

    @device(AIEDevice.npu2)
    def device_body():
        # define types
        uint8_ty = IntegerType.get_unsigned(8)
        int8_ty = IntegerType.get_signless(8)
        int32_ty = IntegerType.get_signless(32)
        uint32_ty = IntegerType.get_unsigned(32)
        # ************************ bneck13 ************************

        ty_bneck_13_layer1_in = MemRefType.get(
            (
                bneck_13_InW1,
                1,
                bneck_13_InC1,
            ),
            int8_ty,
        )
        ty_bneck_13_layer2_in = MemRefType.get(
            (
                bneck_13_InW2,
                1,
                bneck_13_OutC1,
            ),
            uint8_ty,
        )

        # define wts
        # layer1
        ty_bneck_13_layer1_wts_split = MemRefType.get(
            ((bneck_13_InC1 // InputSplit) * (bneck_13_OutC1 // OutputSplit),), int8_ty
        )
        ty_bneck_13_layer1_wts_full = MemRefType.get(
            (bneck_13_InC1 * bneck_13_OutC1,),
            int8_ty,
        )
        # layer2
        b13_layer2_wts_size = 3 * 3 * bneck_13_OutC2 * 1
        ty_bneck_13_layer2_wts = MemRefType.get((3 * 3 * bneck_13_OutC2 * 1,), int8_ty)
        # layer3
        ty_bneck_13_layer3_wts_split = MemRefType.get(
            ((bneck_13_OutC2 // InputSplit) * (bneck_13_OutC3 // OutputSplit2),),
            int8_ty,
        )
        ty_bneck_13_layer3_wts_full = MemRefType.get(
            (bneck_13_OutC2 * bneck_13_OutC3,),
            int8_ty,
        )

        # OUTPUT
        ty_bneck_13_layer1_out = MemRefType.get(
            (
                bneck_13_InW1,
                1,
                bneck_13_OutC1,
            ),
            uint8_ty,
        )
        ty_bneck_13_layer2_out = MemRefType.get(
            (
                bneck_13_InW3,
                1,
                bneck_13_OutC2,
            ),
            uint8_ty,
        )
        ty_bneck_13_layer2_out_split = MemRefType.get(
            (
                bneck_13_InW3,
                1,
                bneck_13_OutC2 // InputSplit,
            ),
            uint8_ty,
        )
        # layer3
        ty_bneck_13_layer3_out = MemRefType.get(
            (
                bneck_13_InW3,
                1,
                bneck_13_OutC3,
            ),
            int8_ty,
        )

        # HERE

        # ************************ bneck14 ************************

        ty_bneck_14_layer1_in = MemRefType.get(
            (
                bneck_14_InW1,
                1,
                bneck_14_InC1,
            ),
            int8_ty,
        )
        ty_bneck_14_layer2_in = MemRefType.get(
            (
                bneck_14_InW2,
                1,
                bneck_14_OutC1,
            ),
            uint8_ty,
        )

        # define wts
        # layer1
        ty_bneck_14_layer1_wts_split = MemRefType.get(
            ((bneck_14_InC1 // InputSplit) * (bneck_14_OutC1 // OutputSplit),), int8_ty
        )
        ty_bneck_14_layer1_wts_full = MemRefType.get(
            (bneck_14_InC1 * bneck_14_OutC1,),
            int8_ty,
        )
        # layer2
        b14_layer2_wts_size = 3 * 3 * bneck_14_OutC2 * 1
        ty_bneck_14_layer2_wts = MemRefType.get((3 * 3 * bneck_14_OutC2 * 1,), int8_ty)
        # layer3
        ty_bneck_14_layer3_wts_split = MemRefType.get(
            ((bneck_14_OutC2 // InputSplit) * (bneck_14_OutC3 // OutputSplit2),),
            int8_ty,
        )
        ty_bneck_14_layer3_wts_full = MemRefType.get(
            (bneck_14_OutC2 * bneck_14_OutC3,),
            int8_ty,
        )

        # OUTPUT
        ty_bneck_14_layer1_out = MemRefType.get(
            (
                bneck_14_InW1,
                1,
                bneck_14_OutC1,
            ),
            uint8_ty,
        )
        ty_bneck_14_layer2_out = MemRefType.get(
            (
                bneck_14_InW3,
                1,
                bneck_14_OutC2,
            ),
            uint8_ty,
        )
        ty_bneck_14_layer2_out_split = MemRefType.get(
            (
                bneck_14_InW3,
                1,
                bneck_14_OutC2 // InputSplit,
            ),
            uint8_ty,
        )
        # layer3
        ty_bneck_14_layer3_out = MemRefType.get(
            (
                bneck_14_InW3,
                1,
                bneck_14_OutC3,
            ),
            int8_ty,
        )

        # Tile declarations
        ShimTile00 = tile(0, 0)
        ShimTile10 = tile(1, 0)
        ShimTile20 = tile(2, 0)
        ShimTile30 = tile(3, 0)

        MemTile01 = tile(0, 1)
        MemTile11 = tile(1, 1)
        MemTile21 = tile(2, 1)
        MemTile31 = tile(3, 1)

        bn13_tile_layer1_put = tile(0, 5)
        # bn13_tile_layer1_get = tile(0, 4)
        bn13_tile_layer1_get = tile(1, 5)
        bn13_tile_layer2 = tile(1, 4)
        # ComputeTile15 = tile(1, 5)

        bn13_tile_layer3_get = tile(1, 3)
        bn13_tile_layer3_put = tile(0, 3)

        cascade_flow(bn13_tile_layer1_put, bn13_tile_layer1_get)
        cascade_flow(bn13_tile_layer3_put, bn13_tile_layer3_get)

        # tiles bn14

        #  conv1
        bn14_tile_layer1_put = tile(1, 2)  # put
        bn14_tile_layer1_get = tile(2, 2)  # get

        cascade_flow(bn14_tile_layer1_put, bn14_tile_layer1_get)

        # conv3
        bn14_tile_layer2 = tile(2, 3)

        # conv
        bn14_tile_layer3_put = tile(2, 5)  # put
        bn14_tile_layer3_get = tile(2, 4)  # get
        # bn14_tile_layer3_get = tile(3, 5)  # get
        cascade_flow(bn14_tile_layer3_put, bn14_tile_layer3_get)

        # Input
        act_in = object_fifo(
            "act_in",
            ShimTile00,
            [bn13_tile_layer1_put, bn13_tile_layer1_get, MemTile01],
            [2, 2, 2, 6],
            ty_bneck_13_layer1_in,
        )
        bn13_skip = object_fifo(
            "bn13_skip", MemTile01, bn13_tile_layer3_get, 2, ty_bneck_13_layer1_in
        )
        object_fifo_link(act_in, bn13_skip)

        # ************ wts ************
        # LAYER1
        bn13_wts_L3L2_layer1 = object_fifo(
            "bn13_wts_L3L2_layer1",
            ShimTile00,
            MemTile01,
            1,
            ty_bneck_13_layer1_wts_full,
        )
        bn13_wts_memtile_layer1_put = object_fifo(
            "bn13_wts_memtile_layer1_put",
            MemTile01,
            bn13_tile_layer1_put,
            [1, 1],
            ty_bneck_13_layer1_wts_split,
        )
        bn13_wts_memtile_layer1_get = object_fifo(
            "bn13_wts_memtile_layer1_get",
            MemTile01,
            bn13_tile_layer1_get,
            [1, 1],
            ty_bneck_13_layer1_wts_split,
        )
        object_fifo_link(
            bn13_wts_L3L2_layer1,
            [bn13_wts_memtile_layer1_put, bn13_wts_memtile_layer1_get],
            [],
            [0, (bneck_13_InC1 * bneck_13_OutC1) // 2],
        )
        bn13_wts_memtile_layer1_put.set_repeat_count(RepeatChannels)
        bn13_wts_memtile_layer1_get.set_repeat_count(RepeatChannels)
        # LAYER2
        # bn13_wts_L3L2_layer2 = object_fifo("bn13_wts_L3L2_layer2", ShimTile10, MemTile11, 1, ty_bneck_13_layer2_wts )
        # bn13_wts_memtile_layer2 = object_fifo("bn13_wts_memtile_layer2",MemTile11,bn13_tile_layer2,1,ty_bneck_13_layer2_wts,)
        # object_fifo_link(bn13_wts_L3L2_layer2, [bn13_wts_memtile_layer2],[],[0])

        bn13_2_wts_ary = np.fromfile(
            data_dir + "bn13_2_chain.txt", sep=",", dtype=np.int8
        )
        bn13_2_wts_static = buffer(
            bn13_tile_layer2,
            np.ndarray[(b13_layer2_wts_size,), np.dtype[np.int8]],
            "bn13_2_wts_static",
            initial_value=bn13_2_wts_ary,
        )

        # LAYER3
        bn13_wts_L3L2_layer3 = object_fifo(
            "bn13_wts_L3L2_layer3",
            ShimTile10,
            MemTile11,
            1,
            ty_bneck_13_layer3_wts_full,
        )
        bn13_wts_memtile_layer3_put = object_fifo(
            "bn13_wts_memtile_layer3_put",
            MemTile11,
            bn13_tile_layer3_put,
            1,
            ty_bneck_13_layer3_wts_split,
        )
        bn13_wts_memtile_layer3_get = object_fifo(
            "bn13_wts_memtile_layer3_get",
            MemTile11,
            bn13_tile_layer3_get,
            1,
            ty_bneck_13_layer3_wts_split,
        )
        object_fifo_link(
            bn13_wts_L3L2_layer3,
            [bn13_wts_memtile_layer3_put, bn13_wts_memtile_layer3_get],
            [],
            [0, (bneck_13_OutC2 * bneck_13_OutC3) // 2],
        )
        bn13_wts_memtile_layer3_put.set_repeat_count(RepeatChannels)
        bn13_wts_memtile_layer3_get.set_repeat_count(RepeatChannels)

        # ************ wts ************
        # wts for new block
        bn14_wts_L3L2_layer1 = object_fifo(
            "bn14_wts_L3L2_layer1",
            ShimTile20,
            MemTile21,
            1,
            ty_bneck_14_layer1_wts_full,
        )
        bn14_wts_memtile_layer1_put = object_fifo(
            "bn14_wts_memtile_layer1_put",
            MemTile21,
            bn14_tile_layer1_put,
            [1, 1],
            ty_bneck_14_layer1_wts_split,
        )
        bn14_wts_memtile_layer1_get = object_fifo(
            "bn14_wts_memtile_layer1_get",
            MemTile21,
            bn14_tile_layer1_get,
            [1, 1],
            ty_bneck_14_layer1_wts_split,
        )
        object_fifo_link(
            bn14_wts_L3L2_layer1,
            [bn14_wts_memtile_layer1_put, bn14_wts_memtile_layer1_get],
            [],
            [0, (bneck_14_InC1 * bneck_14_OutC1) // 2],
        )
        bn14_wts_memtile_layer1_put.set_repeat_count(RepeatChannels)
        bn14_wts_memtile_layer1_get.set_repeat_count(RepeatChannels)
        # LAYER2
        # bn14_wts_L3L2_layer2 = object_fifo("bn14_wts_L3L2_layer2", ShimTile20, MemTile11, 1, ty_bneck_14_layer2_wts)
        # bn14_wts_memtile_layer2 = object_fifo("bn14_wts_memtile_layer2", MemTile11, bn14_tile_layer2, 1, ty_bneck_14_layer2_wts)
        # object_fifo_link(bn14_wts_L3L2_layer2, bn14_wts_memtile_layer2, [], [0])
        bn14_2_wts_ary = np.fromfile(
            data_dir + "bn14_2_chain.txt", sep=",", dtype=np.int8
        )
        bn14_2_wts_static = buffer(
            bn14_tile_layer2,
            np.ndarray[(b14_layer2_wts_size,), np.dtype[np.int8]],
            "bn14_2_wts_static",
            initial_value=bn14_2_wts_ary,
        )

        # LAYER3
        bn14_wts_L3L2_layer3 = object_fifo(
            "bn14_wts_L3L2_layer3",
            ShimTile30,
            MemTile31,
            1,
            ty_bneck_14_layer3_wts_full,
        )
        bn14_wts_memtile_layer3_put = object_fifo(
            "bn14_wts_memtile_layer3_put",
            MemTile31,
            bn14_tile_layer3_put,
            [1, 1],
            ty_bneck_14_layer3_wts_split,
        )
        bn14_wts_memtile_layer3_get = object_fifo(
            "bn14_wts_memtile_layer3_get",
            MemTile31,
            bn14_tile_layer3_get,
            [1, 1],
            ty_bneck_14_layer3_wts_split,
        )
        object_fifo_link(
            bn14_wts_L3L2_layer3,
            [bn14_wts_memtile_layer3_put, bn14_wts_memtile_layer3_get],
            [],
            [0, (bneck_14_OutC2 * bneck_14_OutC3) // 2],
        )
        bn14_wts_memtile_layer3_put.set_repeat_count(RepeatChannels)
        bn14_wts_memtile_layer3_get.set_repeat_count(RepeatChannels)

        act_out = object_fifo(
            "act_out", bn14_tile_layer3_get, ShimTile30, 2, ty_bneck_14_layer3_out
        )

        # Set up compute tiles
        rtp_bn13_tile_layer1_get = buffer(
            bn13_tile_layer1_get,
            np.ndarray[(16,), np.dtype[np.int32]],
            "rtp_bn13_tile_layer1_get",
        )
        rtp_bn13_tile_layer3_get = buffer(
            bn13_tile_layer3_get,
            np.ndarray[(16,), np.dtype[np.int32]],
            "rtp_bn13_tile_layer3_get",
        )

        bottleneckCCore(
            bn13_tile_layer1_put,
            bn13_tile_layer1_get,
            bn13_tile_layer2,
            bn13_tile_layer3_put,
            bn13_tile_layer3_get,
            bn14_tile_layer1_put,
            bn14_tile_layer1_get,
            bn14_tile_layer2,
            bn14_tile_layer3_put,
            bn14_tile_layer3_get,
            bn13_wts_memtile_layer1_put,
            bn13_wts_memtile_layer1_get,
            bn13_2_wts_static,
            bn13_wts_memtile_layer3_put,
            bn13_wts_memtile_layer3_get,
            bn14_wts_memtile_layer1_put,
            bn14_wts_memtile_layer1_get,
            bn14_2_wts_static,
            bn14_wts_memtile_layer3_put,
            bn14_wts_memtile_layer3_get,
            rtp_bn13_tile_layer1_get,
            rtp_bn13_tile_layer3_get,
            bn13_scaleFactor1,
            bn13_scaleFactor2,
            bn13_scaleFactor3,
            bn13_scaleFactorAdd,
            bn14_scaleFactor1,
            bn14_scaleFactor2,
            bn14_scaleFactor3,
            bn14_scaleFactorAdd,
            MemTile21,
            act_in,
            act_out,
            bn13_skip,
        )

        # # instruction stream generation
        activationsInSize32b = (bneck_13_InW1 * bneck_13_InH1 * bneck_13_InC1) // 4
        acitivationsOutSize32b = (bneck_13_InW1 * bneck_13_InH1 * bneck_14_OutC3) // 4

        bneck_13_totalWeightsSize32b_layer1 = (bneck_13_InC1 * bneck_13_OutC1) // 4

        bneck_13_totalWeightsSize32b_layer2 = 0

        bneck_14_totalWeightsSize32b_layer2 = 0

        bneck_13_totalWeightsSize32b_layer3 = (bneck_13_OutC2 * bneck_13_OutC3) // 4

        bneck_13_layer3_offset = (
            bneck_13_totalWeightsSize32b_layer1 + bneck_13_totalWeightsSize32b_layer2
        )

        bneck_13_totalWeightsSize32b_complete = (
            bneck_13_totalWeightsSize32b_layer1
            + bneck_13_totalWeightsSize32b_layer2
            + bneck_13_totalWeightsSize32b_layer3
        )

        bneck_14_layer1_offset = (
            bneck_13_totalWeightsSize32b_layer1
            + bneck_13_totalWeightsSize32b_layer2
            + bneck_13_totalWeightsSize32b_layer3
        )
        bneck_14_layer2_offset = (
            2 * bneck_13_totalWeightsSize32b_layer1
            + bneck_13_totalWeightsSize32b_layer2
            + bneck_13_totalWeightsSize32b_layer3
        )

        bneck_14_layer3_offset = (
            2 * bneck_13_totalWeightsSize32b_layer1
            + bneck_14_totalWeightsSize32b_layer2
            + bneck_13_totalWeightsSize32b_layer3
        )

        totalWeightsSize32b_complete = (
            2
            * (
                bneck_13_totalWeightsSize32b_layer1
                + bneck_13_totalWeightsSize32b_layer3
            )
            + bneck_14_totalWeightsSize32b_layer2
        )

        # TODO Too many tiles causes compilation error?
        # tiles_to_trace = [bn13_tile_layer1_put, bn13_tile_layer1_get, bn13_tile_layer2, bn13_tile_layer3_put, bn13_tile_layer3_get,
        #                   bn14_tile_layer1_put, bn14_tile_layer1_get, bn14_tile_layer2, bn14_tile_layer3_put, bn14_tile_layer3_get,]
        tiles_to_trace = [
            bn13_tile_layer1_put,
            bn13_tile_layer1_get,
            bn13_tile_layer2,
            bn13_tile_layer3_put,
            bn14_tile_layer1_put,
            bn14_tile_layer2,
            bn14_tile_layer3_put,
            bn14_tile_layer3_get,
        ]
        # tiles_to_trace = [bn13_tile_layer1_put]
        # Set up a circuit-switched flow from core to shim for tracing information
        if opts.trace_size > 0:
            trace_utils.configure_tracing_flow(tiles_to_trace, ShimTile00)

        activationsInL3_ty = MemRefType.get((activationsInSize32b,), int32_ty)
        weightsInL3_ty = MemRefType.get((totalWeightsSize32b_complete,), int32_ty)
        activationsOutL3_ty = MemRefType.get((acitivationsOutSize32b,), int32_ty)

        @runtime_sequence(activationsInL3_ty, weightsInL3_ty, activationsOutL3_ty)
        def sequence(inputFromL3, weightsFromL3, outputToL3):
            # NpuWriteRTPOp("rtp04", index=0, value=9)
            # NpuWriteRTPOp("rtp13", index=0, value=11)

            N_in_bytes = bneck_14_InW3 * bneck_14_InH3 * bneck_14_OutC3

            if opts.trace_size > 0:
                trace_utils.configure_tracing_aie2(
                    tiles_to_trace, ShimTile00, opts.trace_size, N_in_bytes
                )
                trace_utils.configure_shim_packet_tracing_aie2(ShimTile00)

            npu_dma_memcpy_nd(
                metadata="act_in",
                bd_id=0,
                mem=inputFromL3,
                sizes=[1, 1, 1, activationsInSize32b],
            )
            npu_dma_memcpy_nd(
                metadata="act_out",
                bd_id=2,
                mem=outputToL3,
                sizes=[1, 1, 1, acitivationsOutSize32b],
            )
            npu_dma_memcpy_nd(
                metadata="bn13_wts_L3L2_layer1",
                bd_id=1,
                mem=weightsFromL3,
                sizes=[1, 1, 1, bneck_13_totalWeightsSize32b_layer1],
            )
            # npu_dma_memcpy_nd(
            #     metadata="bn13_wts_L3L2_layer2",
            #     bd_id=1,
            #     mem=weightsFromL3,
            #     offsets=[0, 0, 0, bneck_13_totalWeightsSize32b_layer1],
            #     sizes=[1, 1, 1, bneck_13_totalWeightsSize32b_layer2],
            # )

            npu_dma_memcpy_nd(
                metadata="bn13_wts_L3L2_layer3",
                bd_id=1,
                mem=weightsFromL3,
                offsets=[0, 0, 0, bneck_13_layer3_offset],
                sizes=[1, 1, 1, bneck_13_totalWeightsSize32b_layer3],
            )

            npu_dma_memcpy_nd(
                metadata="bn14_wts_L3L2_layer1",
                bd_id=1,
                mem=weightsFromL3,
                offsets=[0, 0, 0, bneck_14_layer1_offset],
                sizes=[1, 1, 1, bneck_13_totalWeightsSize32b_layer1],
            )

            # npu_dma_memcpy_nd(
            #     metadata="bn14_wts_L3L2_layer2",
            #     bd_id=1,
            #     mem=weightsFromL3,
            #     offsets=[0, 0, 0, bneck_14_layer2_offset],
            #     sizes=[1, 1, 1, bneck_14_totalWeightsSize32b_layer2],
            # )

            npu_dma_memcpy_nd(
                metadata="bn14_wts_L3L2_layer3",
                bd_id=1,
                mem=weightsFromL3,
                offsets=[0, 0, 0, bneck_14_layer3_offset],
                sizes=[1, 1, 1, bneck_13_totalWeightsSize32b_layer3],
            )

            npu_sync(column=3, row=0, direction=0, channel=0)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "-t",
        "--trace_sz",
        dest="trace_size",
        default=0,
        type=int,
        help="trace size in bytes",
    )
    opts = p.parse_args(sys.argv[1:])

    # Read the existing scale factors
    scale_factor_file = "scale_factors.json"
    data_dir = "data/"
    scale_factors = mb_utils.read_scale_factors(data_dir + scale_factor_file)

    with mlir_mod_ctx() as ctx:
        aie2_bn_13_14(
            bn13_scaleFactor1=scale_factors["BN13"]["conv1x1_1"],
            bn13_scaleFactor2=scale_factors["BN13"]["conv3x3"],
            bn13_scaleFactor3=scale_factors["BN13"]["conv1x1_2"],
            bn13_scaleFactorAdd=scale_factors["BN13"]["skip_add"],
            bn14_scaleFactor1=scale_factors["BN14"]["conv1x1_1"],
            bn14_scaleFactor2=scale_factors["BN14"]["conv3x3"],
            bn14_scaleFactor3=scale_factors["BN14"]["conv1x1_2"],
            bn14_scaleFactorAdd=scale_factors["BN14"]["skip_add"],
        )
        res = ctx.module.operation.verify()
        if res == True:
            print(ctx.module)
        else:
            print(res)
