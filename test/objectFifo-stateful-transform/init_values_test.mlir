//===- init_values_test.mlir ------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK: module @init {
// CHECK:   aie.device(xcve2302) {
// CHECK:     memref.global "public" @of0 : memref<3xi32>
// CHECK:     %tile_1_2 = aie.tile(1, 2)
// CHECK:     %tile_1_3 = aie.tile(1, 3)
// CHECK:     %of0_buff_0 = aie.buffer(%tile_1_2) {sym_name = "of0_buff_0"} : memref<3xi32> = dense<[0, 1, 2]>
// CHECK:     %of0_buff_1 = aie.buffer(%tile_1_2) {sym_name = "of0_buff_1"} : memref<3xi32> = dense<[3, 4, 5]>
// CHECK:     %of0_prod_lock = aie.lock(%tile_1_2, 0) {init = 0 : i32, sym_name = "of0_prod_lock"}
// CHECK:     %of0_cons_lock = aie.lock(%tile_1_2, 1) {init = 2 : i32, sym_name = "of0_cons_lock"}
// CHECK:   }
// CHECK:   aie.device(xcvc1902) {
// CHECK:     memref.global "public" @of3 : memref<2xi32>
// CHECK:     %tile_1_2 = aie.tile(1, 2)
// CHECK:     %tile_1_3 = aie.tile(1, 3)
// CHECK:     %of2_buff_0 = aie.buffer(%tile_1_2) {sym_name = "of2_buff_0"} : memref<2xi32> = dense<[0, 1]>
// CHECK:     %of2_buff_1 = aie.buffer(%tile_1_2) {sym_name = "of2_buff_1"} : memref<2xi32> = dense<[3, 4]>
// CHECK:     %of2_lock_0 = aie.lock(%tile_1_2, 0) {init = 1 : i32, sym_name = "of2_lock_0"}
// CHECK:     %of2_lock_1 = aie.lock(%tile_1_2, 1) {init = 1 : i32, sym_name = "of2_lock_1"}
// CHECK:   }
// CHECK: }

module @init {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)

    aie.objectfifo @of0 (%tile12, {%tile13}, 2 : i32) : !aie.objectfifo<memref<3xi32>> = dense<[[0, 1, 2], [3, 4, 5]]>
 }
//  aie.device(xcve2302) {
//     %tile12 = aie.tile(1, 2)
//     %tile13 = aie.tile(1, 3)

//     aie.objectfifo @of1 (%tile12, {%tile13}, 2 : i32) : !aie.objectfifo<memref<2x2xi32>> = dense<[[[0, 1], [2, 3]], [[4, 5], [6, 7]]]>
//  }
 aie.device(xcvc1902) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)

    aie.objectfifo @of2 (%tile12, {%tile13}, 2 : i32) : !aie.objectfifo<memref<2xi32>> = dense<[[0, 1], [3, 4]]>
 }
}
