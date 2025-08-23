//===- link_via_shared_mem3.mlir --------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK:  aie.device(xcve2302) {
// CHECK:    memref.global "public" @of2 : memref<16xi32>
// CHECK:    memref.global "public" @of1_cons : memref<16xi32>
// CHECK:    memref.global "public" @of1 : memref<16xi32>
// CHECK:    %{{.*}}tile_2_0 = aie.tile(2, 0)
// CHECK:    %{{.*}}tile_1_2 = aie.tile(1, 2)
// CHECK:    %{{.*}}tile_2_2 = aie.tile(2, 2)
// CHECK:    %[[VAL_0:.*]] = aie.buffer(%{{.*}}tile_1_2) {sym_name = "of1_cons_buff_0"} : memref<16xi32>
// CHECK:    %[[VAL_1:.*]] = aie.buffer(%{{.*}}tile_1_2) {sym_name = "of1_cons_buff_1"} : memref<16xi32>
// CHECK:    %[[VAL_2:.*]] = aie.lock(%{{.*}}tile_1_2, 0) {init = 2 : i32, sym_name = "of1_cons_prod_lock_0"}
// CHECK:    %[[VAL_3:.*]] = aie.lock(%{{.*}}tile_1_2, 1) {init = 0 : i32, sym_name = "of1_cons_cons_lock_0"}
// CHECK:    aie.flow(%{{.*}}tile_2_0, DMA : 0, %{{.*}}tile_1_2, DMA : 0)
// CHECK:    %[[VAL_6:.*]] = aie.mem(%{{.*}}tile_1_2) {
// CHECK:      %[[VAL_7:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:    ^bb1:
// CHECK:      aie.use_lock(%[[VAL_2]], AcquireGreaterEqual, 1)
// CHECK:      aie.dma_bd(%[[VAL_0]] : memref<16xi32>, 0, 16)
// CHECK:      aie.use_lock(%[[VAL_3]], Release, 1)
// CHECK:      aie.next_bd ^bb2
// CHECK:    ^bb2:
// CHECK:      aie.use_lock(%[[VAL_2]], AcquireGreaterEqual, 1)
// CHECK:      aie.dma_bd(%[[VAL_1]] : memref<16xi32>, 0, 16)
// CHECK:      aie.use_lock(%[[VAL_3]], Release, 1)
// CHECK:      aie.next_bd ^bb1
// CHECK:    ^bb3:
// CHECK:      aie.end
// CHECK:    }
// CHECK:  }

// In this design, the allocate operation applies to tile_2_2, to which tile_1_2
// does not have direct shared memory access: buffers and locks are created on
// both tiles, following default behaviour of a link.

module @link_AIE2 {
    aie.device(xcve2302) {
        %tile20 = aie.tile(2, 0)
        %tile12 = aie.tile(1, 2)
        %tile22 = aie.tile(2, 2)

        aie.objectfifo @of1 (%tile20, {%tile12}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
        aie.objectfifo @of2 (%tile12, {%tile22}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
        aie.objectfifo.allocate @of2 (%tile22)

        aie.objectfifo.link [@of1] -> [@of2] ([] [])
    }
}
