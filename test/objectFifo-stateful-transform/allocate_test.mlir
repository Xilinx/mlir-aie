//===- allocate_test.mlir ---------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           memref.global "public" @of2 : memref<16xi32>
// CHECK:           memref.global "public" @of1 : memref<16xi32>
// CHECK:           memref.global "public" @of0 : memref<16xi32>
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_1:.*]] = aie.tile(1, 3)
// CHECK:           %[[VAL_2:.*]] = aie.tile(2, 2)
// CHECK:           %of2_buff_0 = aie.buffer(%tile_1_2) {sym_name = "of2_buff_0"} : memref<16xi32> 
// CHECK:           %of2_prod_lock = aie.lock(%tile_1_2, 0) {init = 1 : i32, sym_name = "of2_prod_lock"}
// CHECK:           %of2_cons_lock = aie.lock(%tile_1_2, 1) {init = 0 : i32, sym_name = "of2_cons_lock"}
// CHECK:           %of1_buff_0 = aie.buffer(%tile_1_3) {sym_name = "of1_buff_0"} : memref<16xi32> 
// CHECK:           %of1_prod_lock = aie.lock(%tile_1_3, 2) {init = 1 : i32, sym_name = "of1_prod_lock"}
// CHECK:           %of1_cons_lock = aie.lock(%tile_1_3, 3) {init = 0 : i32, sym_name = "of1_cons_lock"}
// CHECK:           %of0_buff_0 = aie.buffer(%tile_1_3) {sym_name = "of0_buff_0"} : memref<16xi32> 
// CHECK:           %of0_prod_lock = aie.lock(%tile_1_3, 0) {init = 1 : i32, sym_name = "of0_prod_lock"}
// CHECK:           %of0_cons_lock = aie.lock(%tile_1_3, 1) {init = 0 : i32, sym_name = "of0_cons_lock"}
// CHECK:         }

module @same_core {
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
