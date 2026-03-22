#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.

import argparse
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.extras.dialects import memref, arith
from aie.extras.context import mlir_mod_ctx
from aie.extras.dialects.memref import view as memref_view

from aie2_bottleneckBStatic import bottleneckBCoreStatic

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


weights_path = "data/"


def aie2_bn_10_11_12(
    start_row=2,
    start_col=0,
    bn10_scaleFactor1=10,
    bn10_scaleFactor2=7,
    bn10_scaleFactor3=9,
    bn11_scaleFactor1=9,
    bn11_scaleFactor2=8,
    bn11_scaleFactor3=12,
    bn11_scaleFactorAdd=1,
    bn12_scaleFactor1=8,
    bn12_scaleFactor2=8,
    bn12_scaleFactor3=9,
):
    b10_InW1 = 14
    b10_InH1 = 14
    b10_InC1 = 80
    b10_OutC1 = 480

    b10_InW2 = 14
    b10_InH2 = 14
    b10_OutC2 = b10_OutC1

    b10_InW3 = 14
    b10_InH3 = 14
    b10_OutC3 = 112

    b11_OutC1 = 336
    b11_OutC2 = 336
    b11_OutC3 = 112

    b12_OutC1 = 336
    b12_OutC2 = 336
    b12_InW2 = 7
    b12_InH2 = 7
    b12_OutC3 = 80

    enableTrace = False
    trace_size = 16384
    traceSizeInInt32s = trace_size // 4

    b10_layer1_wts_size = b10_InC1 * b10_OutC1
    b10_layer2_wts_size = 3 * 3 * b10_OutC2 * 1
    b10_layer3_wts_size = b10_OutC2 * b10_OutC3

    b11_layer1_wts_size = b10_OutC3 * b11_OutC1
    b11_layer2_wts_size = 3 * 3 * b11_OutC2 * 1
    b11_layer3_wts_size = b11_OutC2 * b11_OutC3

    b12_layer1_wts_size = b11_OutC3 * b12_OutC1
    b12_layer2_wts_size = 3 * 3 * b12_OutC2 * 1
    b12_layer3_wts_size = b12_OutC2 * b12_OutC3

    b12_layer2_3_wts_size = b12_layer2_wts_size + b12_layer3_wts_size

    # @device(AIEDevice.npu1_3col)
    @device(AIEDevice.npu2)
    def device_body():
        # define types
        uint8_ty = IntegerType.get_unsigned(8)
        int8_ty = IntegerType.get_signless(8)
        int32_ty = IntegerType.get_signless(32)
        uint32_ty = IntegerType.get_unsigned(32)

        # Start core
        b_start_col = 0
        b_start_row = 2

        # Get the selected cores
        selected_cores = select_cores(b_start_col, b_start_row)
        # Assign the selected cores to variables
        # bn10_tile_1 = tile(selected_cores[0][0], selected_cores[0][1])
        # bn10_tile_2 = tile(selected_cores[1][0], selected_cores[1][1])
        # bn10_tile_3 = tile(selected_cores[2][0], selected_cores[2][1])

        # Moving to the next group, starting from the 4th core
        # bn11_tile_1 = tile(selected_cores[3][0], selected_cores[3][1])
        # bn11_tile_2 = tile(selected_cores[4][0], selected_cores[4][1])
        # bn11_tile_3 = tile(selected_cores[5][0], selected_cores[5][1])

        # Moving to the next group, starting from the 7th core
        # bn12_tile_1 = tile(selected_cores[6][0], selected_cores[6][1])
        # bn12_tile_2 = tile(selected_cores[7][0], selected_cores[7][1])
        # bn12_tile_3 = tile(selected_cores[8][0], selected_cores[8][1])

        bn10_tile_1 = tile(1, 5)
        bn10_tile_2 = tile(2, 4)
        bn10_tile_3 = tile(2, 5)

        bn11_tile_1 = tile(3, 2)
        bn11_tile_2 = tile(3, 4)
        bn11_tile_3 = tile(2, 2)

        bn12_tile_1 = tile(3, 5)
        bn12_tile_2 = tile(4, 4)

        ShimTile00 = tile(0, 0)
        ShimTile10 = tile(1, 0)

        MemTile01 = tile(0, 1)
        MemTile11 = tile(1, 1)
        MemTile21 = tile(2, 1)

        b10_layer1_in = MemRefType.get(
            (
                b10_InW1,
                1,
                b10_InC1,
            ),
            int8_ty,
        )
        b12_layer3_out = MemRefType.get(
            (
                b12_InW2,
                1,
                b12_OutC3,
            ),
            int8_ty,
        )
        # define wts
        b10_layer1_wts = MemRefType.get((b10_InC1 * b10_OutC1,), int8_ty)
        b10_layer2_wts = MemRefType.get((3 * 3 * b10_OutC2 * 1,), int8_ty)
        b10_layer3_wts = MemRefType.get((b10_OutC2 * b10_OutC3,), int8_ty)
        b10_all_wts = MemRefType.get(
            (b10_InC1 * b10_OutC1 + 3 * 3 * b10_OutC2 * 1 + b10_OutC2 * b10_OutC3,),
            int8_ty,
        )
        # output
        b10_layer1_out = MemRefType.get(
            (
                b10_InW2,
                1,
                b10_OutC1,
            ),
            uint8_ty,
        )
        b10_layer2_out = MemRefType.get(
            (
                b10_InW3,
                1,
                b10_OutC2,
            ),
            uint8_ty,
        )
        b10_layer3_out = MemRefType.get(
            (
                b10_InW3,
                1,
                b10_OutC3,
            ),
            int8_ty,
        )
        # ************************ bneck11 ************************
        # input
        b11_layer1_in = MemRefType.get(
            (
                b10_InW3,
                1,
                b10_OutC3,
            ),
            int8_ty,
        )
        b11_layer2_in = MemRefType.get(
            (
                b10_InW3,
                1,
                b11_OutC1,
            ),
            uint8_ty,
        )
        b11_layer3_in = MemRefType.get(
            (
                b10_InW3,
                1,
                b11_OutC2,
            ),
            uint8_ty,
        )

        # define wts
        b11_layer1_wts = MemRefType.get((b10_OutC3 * b11_OutC1,), int8_ty)
        b11_layer2_wts = MemRefType.get((3 * 3 * b11_OutC2 * 1,), int8_ty)
        b11_layer3_wts = MemRefType.get((b11_OutC2 * b11_OutC3,), int8_ty)
        b11_all_wts = MemRefType.get(
            (b10_OutC3 * b11_OutC1 + 3 * 3 * b11_OutC2 * 1 + b11_OutC2 * b11_OutC3,),
            int8_ty,
        )
        # output
        b11_layer1_out = MemRefType.get(
            (
                b10_InW3,
                1,
                b11_OutC1,
            ),
            uint8_ty,
        )
        b11_layer2_out = MemRefType.get(
            (
                b10_InW3,
                1,
                b11_OutC2,
            ),
            uint8_ty,
        )
        b11_layer3_out = MemRefType.get(
            (
                b10_InW3,
                1,
                b11_OutC3,
            ),
            int8_ty,
        )
        # ************************ bneck12 ************************
        b12_layer1_in = MemRefType.get(
            (
                b10_InW1,
                1,
                b11_OutC3,
            ),
            int8_ty,
        )
        b12_layer2_in = MemRefType.get(
            (
                b10_InW1,
                1,
                b12_OutC1,
            ),
            uint8_ty,
        )
        b12_layer3_in = MemRefType.get(
            (
                b12_InW2,
                1,
                b12_OutC2,
            ),
            uint8_ty,
        )
        # define wts
        b12_layer1_wts = MemRefType.get((b11_OutC3 * b12_OutC1,), int8_ty)
        b12_layer2_wts = MemRefType.get((3 * 3 * b12_OutC2 * 1,), int8_ty)
        b12_layer3_wts = MemRefType.get((b12_OutC2 * b12_OutC3,), int8_ty)
        b12_all_wts = MemRefType.get(
            (b11_OutC3 * b12_OutC1 + 3 * 3 * b12_OutC2 * 1 + b12_OutC2 * b12_OutC3,),
            int8_ty,
        )
        # output
        b12_layer1_out = MemRefType.get(
            (
                b10_InW3,
                1,
                b12_OutC1,
            ),
            uint8_ty,
        )
        b12_layer2_out = MemRefType.get(
            (
                b12_InW2,
                1,
                b12_OutC2,
            ),
            uint8_ty,
        )
        b12_layer3_out = MemRefType.get(
            (
                b12_InW2,
                1,
                b12_OutC3,
            ),
            int8_ty,
        )
        # Input
        act_in = object_fifo("act_in", ShimTile00, bn10_tile_1, 2, b10_layer1_in)
        # act_out = object_fifo("act_out", bn12_tile_3, ShimTile10, 2, b12_layer3_out)
        # act_out = object_fifo("act_out", bn12_tile_2, ShimTile10, 2, b12_layer2_out)
        act_out = object_fifo("act_out", bn12_tile_2, ShimTile10, 2, b12_layer3_out)

        # wts
        bn10_1_wts_ary = np.fromfile(
            weights_path + "bn10_1_chain.txt", sep=",", dtype=np.int8
        )
        bn10_2_wts_ary = np.fromfile(
            weights_path + "bn10_2_chain.txt", sep=",", dtype=np.int8
        )
        bn10_3_wts_ary = np.fromfile(
            weights_path + "bn10_3_chain.txt", sep=",", dtype=np.int8
        )

        bn11_1_wts_ary = np.fromfile(
            weights_path + "bn11_1_chain.txt", sep=",", dtype=np.int8
        )
        bn11_2_wts_ary = np.fromfile(
            weights_path + "bn11_2_chain.txt", sep=",", dtype=np.int8
        )
        bn11_3_wts_ary = np.fromfile(
            weights_path + "bn11_3_chain.txt", sep=",", dtype=np.int8
        )

        bn12_1_wts_ary = np.fromfile(
            weights_path + "bn12_1_chain.txt", sep=",", dtype=np.int8
        )
        # bn12_2_wts_ary=np.fromfile(weights_path+"bn12_2_chain.txt", sep=",", dtype=np.int8)
        # bn12_3_wts_ary=np.fromfile(weights_path+"bn12_3_chain.txt", sep=",", dtype=np.int8)
        bn12_2_3_wts_ary = np.fromfile(
            weights_path + "bn12_2_3_chain.txt", sep=",", dtype=np.int8
        )

        bn10_1_wts_static = buffer(
            bn10_tile_1,
            np.ndarray[(b10_layer1_wts_size,), np.dtype[np.int8]],
            "bn10_1_wts_static",
            initial_value=bn10_1_wts_ary,
        )
        bn10_2_wts_static = buffer(
            bn10_tile_2,
            np.ndarray[(b10_layer2_wts_size,), np.dtype[np.int8]],
            "bn10_2_wts_static",
            initial_value=bn10_2_wts_ary,
        )
        bn10_3_wts_static = buffer(
            bn10_tile_3,
            np.ndarray[(b10_layer3_wts_size,), np.dtype[np.int8]],
            "bn10_3_wts_static",
            initial_value=bn10_3_wts_ary,
        )

        bn11_1_wts_static = buffer(
            bn11_tile_1,
            np.ndarray[(b11_layer1_wts_size,), np.dtype[np.int8]],
            "bn11_1_wts_static",
            initial_value=bn11_1_wts_ary,
        )
        bn11_2_wts_static = buffer(
            bn11_tile_2,
            np.ndarray[(b11_layer2_wts_size,), np.dtype[np.int8]],
            "bn11_2_wts_static",
            initial_value=bn11_2_wts_ary,
        )
        bn11_3_wts_static = buffer(
            bn11_tile_3,
            np.ndarray[(b11_layer3_wts_size,), np.dtype[np.int8]],
            "bn11_3_wts_static",
            initial_value=bn11_3_wts_ary,
        )

        bn12_1_wts_static = buffer(
            bn12_tile_1,
            np.ndarray[(b12_layer1_wts_size,), np.dtype[np.int8]],
            "bn12_1_wts_static",
            initial_value=bn12_1_wts_ary,
        )
        bn12_2_3_wts_static = buffer(
            bn12_tile_2,
            np.ndarray[(b12_layer2_3_wts_size,), np.dtype[np.int8]],
            "bn12_2_3_wts_static",
            initial_value=bn12_2_3_wts_ary,
        )

        # # wts
        # wts_b10_L3L2 = object_fifo(
        #     "wts_b10_L3L2", ShimTile00, MemTile01, 1, b10_all_wts
        # )
        # bn10_1_wts_OF_L3L1 = object_fifo(
        #     "weightsInBN10_layer1", MemTile01, bn10_tile_1, 1, b10_layer1_wts
        # )
        # bn10_2_wts_OF_L3L1 = object_fifo(
        #     "weightsInBN10_layer2",
        #     MemTile01,
        #     bn10_tile_2,
        #     1,
        #     b10_layer2_wts,
        # )
        # bn10_3_wts_OF_L3L1 = object_fifo(
        #     "weightsInBN10_layer3",
        #     MemTile01,
        #     bn10_tile_3,
        #     1,
        #     b10_layer3_wts,
        # )
        # object_fifo_link(
        #     wts_b10_L3L2,
        #     [bn10_1_wts_OF_L3L1, bn10_2_wts_OF_L3L1, bn10_3_wts_OF_L3L1],
        #     [],
        #     [0, b10_InC1 * b10_OutC1, b10_InC1 * b10_OutC1 + 3 * 3 * b10_OutC2 * 1],
        # )

        # # wts
        # wts_b11_L3L2 = object_fifo(
        #     "wts_b11_L3L2", ShimTile10, MemTile11, 1, b11_all_wts
        # )
        # bn11_1_wts_OF_L3L1 = object_fifo(
        #     "weightsInBN11_layer1", MemTile11, bn11_tile_1, 1, b11_layer1_wts
        # )
        # bn11_2_wts_OF_L3L1 = object_fifo(
        #     "weightsInBN11_layer2",
        #     MemTile11,
        #     bn11_tile_2,
        #     1,
        #     b11_layer2_wts,
        # )
        # bn11_3_wts_OF_L3L1 = object_fifo(
        #     "weightsInBN11_layer3",
        #     MemTile11,
        #     bn11_tile_3,
        #     1,
        #     b11_layer3_wts,
        # )
        # object_fifo_link(
        #     wts_b11_L3L2,
        #     [bn11_1_wts_OF_L3L1, bn11_2_wts_OF_L3L1, bn11_3_wts_OF_L3L1],
        #     [],
        #     [0, b10_OutC3 * b11_OutC1, b10_OutC3 * b11_OutC1 + 3 * 3 * b11_OutC2 * 1],
        # )

        # # # wts
        # wts_b12_L3L2 = object_fifo(
        #     "wts_b12_L3L2", ShimTile10, MemTile21, 1, b12_all_wts
        # )
        # bn12_1_wts_OF_L3L1 = object_fifo(
        #     "weightsInBN12_layer1", MemTile21, bn12_tile_1, 1, b12_layer1_wts
        # )
        # bn12_2_wts_OF_L3L1 = object_fifo(
        #     "weightsInBN12_layer2",
        #     MemTile21,
        #     bn12_tile_2,
        #     1,
        #     b12_layer2_wts,
        # )
        # bn12_3_wts_OF_L3L1 = object_fifo(
        #     "weightsInBN12_layer3",
        #     MemTile21,
        #     bn12_tile_3,
        #     1,
        #     b12_layer3_wts,
        # )
        # object_fifo_link(
        #     wts_b12_L3L2,
        #     [bn12_1_wts_OF_L3L1, bn12_2_wts_OF_L3L1, bn12_3_wts_OF_L3L1],
        #     [],
        #     [0, b11_OutC3 * b12_OutC1, b11_OutC3 * b12_OutC1 + 3 * 3 * b12_OutC2 * 1],
        # )

        bn10_1_rtp = buffer(
            bn10_tile_1, np.ndarray[(16,), np.dtype[np.int32]], "bn10_1_rtp"
        )
        bn10_2_rtp = buffer(
            bn10_tile_2, np.ndarray[(16,), np.dtype[np.int32]], "bn10_2_rtp"
        )
        bn10_3_rtp = buffer(
            bn10_tile_3, np.ndarray[(16,), np.dtype[np.int32]], "bn10_3_rtp"
        )

        bn11_1_rtp = buffer(
            bn11_tile_1, np.ndarray[(16,), np.dtype[np.int32]], "bn11_1_rtp"
        )
        bn11_2_rtp = buffer(
            bn11_tile_2, np.ndarray[(16,), np.dtype[np.int32]], "bn11_2_rtp"
        )
        bn11_3_rtp = buffer(
            bn11_tile_3, np.ndarray[(16,), np.dtype[np.int32]], "bn11_3_rtp"
        )

        bn12_1_rtp = buffer(
            bn12_tile_1, np.ndarray[(16,), np.dtype[np.int32]], "bn12_1_rtp"
        )
        bn12_2_rtp = buffer(
            bn12_tile_2, np.ndarray[(16,), np.dtype[np.int32]], "bn12_2_rtp"
        )
        # TODO
        # bn12_3_rtp = buffer(
        #     bn12_tile_3, np.ndarray[(16,), np.dtype[np.int32]], "bn12_3_rtp"
        # )

        bottleneckBCoreStatic(
            "B",
            bn10_tile_1,
            bn10_tile_2,
            bn10_tile_3,
            bn11_tile_1,
            bn11_tile_2,
            bn11_tile_3,
            bn12_tile_1,
            bn12_tile_2,
            # bn12_tile_3,
            # bn10_1_wts_OF_L3L1,
            # bn10_2_wts_OF_L3L1,
            # bn10_3_wts_OF_L3L1,
            # bn11_1_wts_OF_L3L1,
            # bn11_2_wts_OF_L3L1,
            # bn11_3_wts_OF_L3L1,
            # bn12_1_wts_OF_L3L1,
            # bn12_2_wts_OF_L3L1,
            # bn12_3_wts_OF_L3L1,
            bn10_1_wts_static,
            bn10_2_wts_static,
            bn10_3_wts_static,
            bn11_1_wts_static,
            bn11_2_wts_static,
            bn11_3_wts_static,
            bn12_1_wts_static,
            bn12_2_3_wts_static,
            b12_layer2_out,
            bn10_1_rtp,
            bn10_2_rtp,
            bn10_3_rtp,
            bn11_1_rtp,
            bn11_2_rtp,
            bn11_3_rtp,
            bn12_1_rtp,
            bn12_2_rtp,
            # bn12_3_rtp,
            MemTile01,
            act_in,
            act_out,
            bn10_scaleFactor1,
            bn10_scaleFactor2,
            bn10_scaleFactor3,
            bn11_scaleFactor1,
            bn11_scaleFactor2,
            bn11_scaleFactor3,
            bn11_scaleFactorAdd,
            bn12_scaleFactor1,
            bn12_scaleFactor2,
            bn12_scaleFactor3,
        )

        # # instruction stream generation
        activationsInSize32b = (b10_InW1 * b10_InH1 * b10_InC1) // 4
        # acitivationsOutSize32b = (b12_InW2 * b12_InH2 * b12_OutC3) // 4
        acitivationsOutSize32b = (b12_InW2 * b12_InW2 * b12_OutC3) // 4

        bn10_totalWeightsSize32b = (
            b10_InC1 * b10_OutC1 + 3 * 3 * b10_OutC2 * 1 + b10_OutC2 * b10_OutC3
        ) // 4

        bn11_totalWeightsSize32b = (
            b10_OutC3 * b11_OutC1 + 3 * 3 * b11_OutC2 * 1 + b11_OutC2 * b11_OutC3
        ) // 4

        bn12_totalWeightsSize32b = (
            b11_OutC3 * b12_OutC1 + 3 * 3 * b12_OutC2 * 1 + b12_OutC2 * b12_OutC3
        ) // 4

        bn12_Offset_32b = bn10_totalWeightsSize32b + bn11_totalWeightsSize32b

        totalWeightsSize32b_complete = (
            bn10_totalWeightsSize32b
            + bn11_totalWeightsSize32b
            + bn12_totalWeightsSize32b
        )

        activationsInL3_ty = MemRefType.get((activationsInSize32b,), int32_ty)
        weightsInL3_ty = MemRefType.get((totalWeightsSize32b_complete,), int32_ty)
        activationsOutL3_ty = MemRefType.get((acitivationsOutSize32b,), int32_ty)

        @runtime_sequence(activationsInL3_ty, weightsInL3_ty, activationsOutL3_ty)
        def sequence(inputFromL3, weightsFromL3, outputToL3):
            NpuWriteRTPOp("bn10_1_rtp", index=0, value=bn10_scaleFactor1)
            NpuWriteRTPOp("bn10_2_rtp", index=0, value=bn10_scaleFactor2)
            NpuWriteRTPOp("bn10_3_rtp", index=0, value=bn10_scaleFactor3)

            NpuWriteRTPOp("bn11_1_rtp", index=0, value=bn11_scaleFactor1)
            NpuWriteRTPOp("bn11_2_rtp", index=0, value=bn11_scaleFactor2)
            NpuWriteRTPOp("bn11_3_rtp", index=0, value=bn11_scaleFactor3)
            NpuWriteRTPOp("bn11_3_rtp", index=1, value=bn11_scaleFactorAdd)

            NpuWriteRTPOp("bn12_1_rtp", index=0, value=bn12_scaleFactor1)
            NpuWriteRTPOp("bn12_2_rtp", index=0, value=bn12_scaleFactor2)
            NpuWriteRTPOp("bn12_2_rtp", index=1, value=bn12_scaleFactor3)
            # NpuWriteRTPOp("bn12_3_rtp", index=0, value=bn12_scaleFactor3)

            npu_dma_memcpy_nd(
                metadata="act_in",
                bd_id=0,
                mem=inputFromL3,
                sizes=[1, 1, 1, activationsInSize32b],
            )
            npu_dma_memcpy_nd(
                metadata="act_out",
                # metadata="b12_layer2_out",
                bd_id=2,
                mem=outputToL3,
                sizes=[1, 1, 1, acitivationsOutSize32b],
            )
            # npu_dma_memcpy_nd(
            #     metadata="wts_b10_L3L2",
            #     bd_id=1,
            #     mem=weightsFromL3,
            #     sizes=[1, 1, 1, bn10_totalWeightsSize32b],
            # )
            # npu_dma_memcpy_nd(
            #     metadata="wts_b11_L3L2",
            #     bd_id=1,
            #     mem=weightsFromL3,
            #     offsets=[0, 0, 0, bn10_totalWeightsSize32b],
            #     sizes=[1, 1, 1, bn11_totalWeightsSize32b],
            # )
            # npu_dma_memcpy_nd(
            #     metadata="wts_b12_L3L2",
            #     bd_id=1,
            #     mem=weightsFromL3,
            #     offsets=[0, 0, 0, bn12_Offset_32b],
            #     sizes=[1, 1, 1, bn12_totalWeightsSize32b],
            # )
            npu_sync(column=1, row=0, direction=0, channel=0)


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
        aie2_bn_10_11_12(
            bn10_scaleFactor1=scale_factors["BN10"]["conv1x1_1"],
            bn10_scaleFactor2=scale_factors["BN10"]["conv3x3"],
            bn10_scaleFactor3=scale_factors["BN10"]["conv1x1_2"],
            bn11_scaleFactor1=scale_factors["BN11"]["conv1x1_1"],
            bn11_scaleFactor2=scale_factors["BN11"]["conv3x3"],
            bn11_scaleFactor3=scale_factors["BN11"]["conv1x1_2"],
            bn11_scaleFactorAdd=scale_factors["BN11"]["skip_add"],
            bn12_scaleFactor1=scale_factors["BN12"]["conv1x1_1"],
            bn12_scaleFactor2=scale_factors["BN12"]["conv3x3"],
            bn12_scaleFactor3=scale_factors["BN12"]["conv1x1_2"],
        )

        res = ctx.module.operation.verify()
        if res == True:
            print(ctx.module)
        else:
            print(res)
