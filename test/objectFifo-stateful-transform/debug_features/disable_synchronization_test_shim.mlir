//===- disable_synchronization_test_shim.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(2, 0)
// CHECK:           %[[VAL_1:.*]] = aie.tile(1, 3)
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of0_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of0_cons_buff_1"} : memref<16xi32>
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           %[[VAL_4:.*]] = aie.external_buffer {sym_name = "ext_buffer_in"} : memref<64xi32>
// CHECK:           aie.shim_dma_allocation @of0_shim_alloc(%[[VAL_0]], MM2S, 0)
// CHECK:           %[[VAL_5:.*]] = aie.shim_dma(%[[VAL_0]]) {
// CHECK:             %[[VAL_6:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:
// CHECK:             aie.dma_bd(%[[VAL_4]] : memref<64xi32> offset = 0 len = 64)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_7:.*]] = aie.mem(%[[VAL_1]]) {
// CHECK:             %[[VAL_8:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @disable_sync {
 aie.device(xcve2302) {
    %tile20 = aie.tile(2, 0)
    %tile13 = aie.tile(1, 3)

    aie.objectfifo @of0 (%tile20, {%tile13}, 2 : i32) { disable_synchronization = true } : !aie.objectfifo<memref<16xi32>>

    %ext_buffer_in = aie.external_buffer {sym_name = "ext_buffer_in"}: memref<64xi32>
    aie.objectfifo.register_external_buffers @of0 (%tile20, {%ext_buffer_in}) : (memref<64xi32>)
 }
}
