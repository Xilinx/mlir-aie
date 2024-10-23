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
// CHECK:           memref.global "public" @of1 : memref<16xi32>
// CHECK:           memref.global "public" @of0 : memref<16xi32>
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_1:.*]] = aie.tile(1, 3)
// CHECK:           %of1_buff_0 = aie.buffer(%[[VAL_1]]) {sym_name = "of1_buff_0"} : memref<16xi32> 
// CHECK:           %of1_buff_1 = aie.buffer(%[[VAL_1]]) {sym_name = "of1_buff_1"} : memref<16xi32> 
// CHECK:           %of1_prod_lock = aie.lock(%[[VAL_1]], 2) {init = 2 : i32, sym_name = "of1_prod_lock"}
// CHECK:           %of1_cons_lock = aie.lock(%[[VAL_1]], 3) {init = 0 : i32, sym_name = "of1_cons_lock"}
// CHECK:           %of0_buff_0 = aie.buffer(%[[VAL_1]]) {sym_name = "of0_buff_0"} : memref<16xi32> 
// CHECK:           %of0_prod_lock = aie.lock(%[[VAL_1]], 0) {init = 1 : i32, sym_name = "of0_prod_lock"}
// CHECK:           %of0_cons_lock = aie.lock(%[[VAL_1]], 1) {init = 0 : i32, sym_name = "of0_cons_lock"}
// CHECK:         }

module @same_core {
    aie.device(xcve2302) {
        %tile12 = aie.tile(1, 2)
        %tile13 = aie.tile(1, 3)

        aie.objectfifo @of0 (%tile12, {%tile12}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
        aie.objectfifo.allocate @of0 (%tile13)

        aie.objectfifo @of1 (%tile12, {%tile13}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
        aie.objectfifo.allocate @of1 (%tile13)
    }
}
