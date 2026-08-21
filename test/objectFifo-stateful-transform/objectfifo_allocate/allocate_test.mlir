//===- allocate_test.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK-DAG:           %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK-DAG:           %[[VAL_1:.*]] = aie.tile(1, 3)
// CHECK-DAG:           %of2_buff_0 = aie.buffer(%tile_1_2) {sym_name = "of2_buff_0"} : memref<16xi32>
// CHECK-DAG:           %of2_prod_lock_0 = aie.lock(%tile_1_2) {init = 1 : i32, sym_name = "of2_prod_lock_0"}
// CHECK-DAG:           %of2_cons_lock_0 = aie.lock(%tile_1_2) {init = 0 : i32, sym_name = "of2_cons_lock_0"}
// CHECK-DAG:           %of1_buff_0 = aie.buffer(%tile_1_3) {sym_name = "of1_buff_0"} : memref<16xi32>
// CHECK-DAG:           %of1_prod_lock_0 = aie.lock(%tile_1_3) {init = 1 : i32, sym_name = "of1_prod_lock_0"}
// CHECK-DAG:           %of1_cons_lock_0 = aie.lock(%tile_1_3) {init = 0 : i32, sym_name = "of1_cons_lock_0"}
// CHECK-DAG:           %of0_buff_0 = aie.buffer(%tile_1_3) {sym_name = "of0_buff_0"} : memref<16xi32>
// CHECK-DAG:           %of0_prod_lock_0 = aie.lock(%tile_1_3) {init = 1 : i32, sym_name = "of0_prod_lock_0"}
// CHECK-DAG:           %of0_cons_lock_0 = aie.lock(%tile_1_3) {init = 0 : i32, sym_name = "of0_cons_lock_0"}
// CHECK:         }

module @allocate {
    aie.device(xcve2302) {
        %tile12 = aie.tile(1, 2)
        %tile13 = aie.tile(1, 3)
        %tile22 = aie.tile(2, 2)

        aie.objectfifo @of0 (%tile12, {%tile12}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
        aie.objectfifo.allocate @of0 (%tile13)

        aie.objectfifo @of1 (%tile12, {%tile13}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
        aie.objectfifo.allocate @of1 (%tile13)

        aie.objectfifo @of2 (%tile12, {%tile22}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
        aie.objectfifo.allocate @of2 (%tile12)
    }
}
