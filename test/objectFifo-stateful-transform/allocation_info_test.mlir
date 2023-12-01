//===- allocation_info_test.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
// Date: May 20th 2023
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   AIE.device(xcve2302) {
// CHECK:           %[[VAL_0:.*]] = AIE.tile(2, 0)
// CHECK:           %[[VAL_1:.*]] = AIE.tile(2, 2)
// CHECK:           %[[VAL_2:.*]] = AIE.tile(2, 3)
// CHECK:           AIE.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           AIE.flow(%[[VAL_1]], DMA : 0, %[[VAL_0]], DMA : 0)
// CHECK:           AIE.flow(%[[VAL_0]], DMA : 1, %[[VAL_2]], DMA : 0)
// CHECK:           AIE.flow(%[[VAL_2]], DMA : 0, %[[VAL_0]], DMA : 1)
// CHECK:           AIE.shimDMAAllocation @of_in_0(MM2S, 0, 2)
// CHECK:           AIE.shimDMAAllocation @of_out_0(S2MM, 0, 2)
// CHECK:           AIE.shimDMAAllocation @of_in_1(MM2S, 1, 2)
// CHECK:           AIE.shimDMAAllocation @of_out_1(S2MM, 1, 2)
// CHECK:         }

module @alloc {
    AIE.device(xcve2302) {
        %tile20 = AIE.tile(2, 0)
        %tile22 = AIE.tile(2, 2)
        %tile23 = AIE.tile(2, 3)

        AIE.objectfifo @of_in_0 (%tile20, {%tile22}, 2 : i32) : !AIE.objectfifo<memref<64xi16>>
        AIE.objectfifo @of_out_0 (%tile22, {%tile20}, 2 : i32) : !AIE.objectfifo<memref<64xi16>>

        AIE.objectfifo @of_in_1 (%tile20, {%tile23}, 2 : i32) : !AIE.objectfifo<memref<64xi16>>
        AIE.objectfifo @of_out_1 (%tile23, {%tile20}, 2 : i32) : !AIE.objectfifo<memref<64xi16>>
    }
}
