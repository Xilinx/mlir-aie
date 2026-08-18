//===- link_test_DDR_to_L1.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Date: May 9th 2023
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(2, 0)
// CHECK:           %[[VAL_1:.*]] = aie.lock(%[[VAL_0]]) {init = 1 : i32, sym_name = "to_memTile_prod_lock_0"}
// CHECK:           %[[VAL_2:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "to_memTile_cons_lock_0"}
// CHECK:           %[[VAL_3:.*]] = aie.tile(2, 1)
// CHECK:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "to_memTile_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_5:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "to_memTile_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_3]]) {init = 2 : i32, sym_name = "to_memTile_cons_prod_lock_0"}
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_3]]) {init = 0 : i32, sym_name = "to_memTile_cons_cons_lock_0"}
// CHECK:           %[[VAL_8:.*]] = aie.tile(2, 2)
// CHECK:           %[[VAL_9:.*]] = aie.buffer(%[[VAL_8]]) {sym_name = "from_memTile_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_10:.*]] = aie.buffer(%[[VAL_8]]) {sym_name = "from_memTile_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_11:.*]] = aie.lock(%[[VAL_8]]) {init = 2 : i32, sym_name = "from_memTile_cons_prod_lock_0"}
// CHECK:           %[[VAL_12:.*]] = aie.lock(%[[VAL_8]]) {init = 0 : i32, sym_name = "from_memTile_cons_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_3]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_3]], DMA : 0, %[[VAL_8]], DMA : 0)
// CHECK:           %[[VAL_13:.*]] = aie.external_buffer {sym_name = "ext_buff_in"} : memref<16xi32>
// CHECK:           aie.shim_dma_allocation @to_memTile_shim_alloc(%[[VAL_0]], MM2S, 0)
// CHECK:           %[[VAL_14:.*]] = aie.shim_dma(%[[VAL_0]]) {
// CHECK:             %[[VAL_15:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_16:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_2]], AcquireGreaterEqual, %[[VAL_15]])
// CHECK:             aie.dma_bd(%[[VAL_13]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_1]], Release, %[[VAL_15]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_17:.*]] = aie.memtile_dma(%[[VAL_3]]) {
// CHECK:             %[[VAL_18:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_19:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_18]])
// CHECK:             aie.dma_bd(%[[VAL_4]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_18]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_18]])
// CHECK:             aie.dma_bd(%[[VAL_5]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_18]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_20:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, %[[VAL_18]])
// CHECK:             aie.dma_bd(%[[VAL_4]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[VAL_18]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, %[[VAL_18]])
// CHECK:             aie.dma_bd(%[[VAL_5]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[VAL_18]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_21:.*]] = aie.mem(%[[VAL_8]]) {
// CHECK:             %[[VAL_22:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_23:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, %[[VAL_22]])
// CHECK:             aie.dma_bd(%[[VAL_9]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_12]], Release, %[[VAL_22]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, %[[VAL_22]])
// CHECK:             aie.dma_bd(%[[VAL_10]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_12]], Release, %[[VAL_22]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @link_DDR_L1 {
    aie.device(xcve2302) {
        %tile20 = aie.tile(2, 0)
        %tile21 = aie.tile(2, 1)
        %tile22 = aie.tile(2, 2)

        aie.objectfifo @to_memTile (%tile20, {%tile21}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
        aie.objectfifo @from_memTile (%tile21, {%tile22}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

        aie.objectfifo.link [@to_memTile] -> [@from_memTile] ([] [])

        %ext_buff_in = aie.external_buffer {sym_name = "ext_buff_in"}: memref<16xi32>
        aie.objectfifo.register_external_buffers @to_memTile (%tile20, {%ext_buff_in}) : (memref<16xi32>)
    }
}
