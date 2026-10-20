//===- allocate_bridge_test.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// Tiles (0, 2) and (0, 4) share no memory, but both reach (0, 3)'s: the
// delegate holds of0's objects and locks, and no DMA or flow moves them.

// CHECK-LABEL:   aie.device(npu2) {
// CHECK-DAG:       %of0_buff_0 = aie.buffer(%tile_0_3) {sym_name = "of0_buff_0"} : memref<16xi32>
// CHECK-DAG:       %of0_buff_1 = aie.buffer(%tile_0_3) {sym_name = "of0_buff_1"} : memref<16xi32>
// CHECK-DAG:       %of0_prod_lock_0 = aie.lock(%tile_0_3) {init = 2 : i32, sym_name = "of0_prod_lock_0"}
// CHECK-DAG:       %of0_cons_lock_0 = aie.lock(%tile_0_3) {init = 0 : i32, sym_name = "of0_cons_lock_0"}
// CHECK-NOT:       aie.flow
// CHECK-NOT:       aie.mem(
// CHECK:           aie.core(%tile_0_2)
// CHECK:             aie.use_lock(%of0_prod_lock_0, AcquireGreaterEqual, %{{.*}})
// CHECK:             aie.use_lock(%of0_cons_lock_0, Release, %{{.*}})
// CHECK:           aie.core(%tile_0_4)
// CHECK:             aie.use_lock(%of0_cons_lock_0, AcquireGreaterEqual, %{{.*}})
// CHECK:             aie.use_lock(%of0_prod_lock_0, Release, %{{.*}})

module @allocate_bridge {
  aie.device(npu2) {
    %tile02 = aie.tile(0, 2)
    %tile03 = aie.tile(0, 3)
    %tile04 = aie.tile(0, 4)

    aie.objectfifo @of0 (%tile02, {%tile04}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.allocate @of0 (%tile03)

    %core02 = aie.core(%tile02) {
      %0 = aie.objectfifo.acquire @of0(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @of0 (Produce, 1)
      aie.end
    }
    %core04 = aie.core(%tile04) {
      %0 = aie.objectfifo.acquire @of0(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @of0 (Consume, 1)
      aie.end
    }
  }
}
