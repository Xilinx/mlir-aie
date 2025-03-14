//===- via_shared_mem_test.mlir --------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK: module @viaSharedMem {
// CHECK:   aie.device(npu1) {
// CHECK:     memref.global "public" @of1 : memref<16xi32>
// CHECK:     memref.global "public" @of0 : memref<16xi32>
// CHECK:     %{{.*}}tile_1_2 = aie.tile(1, 2)
// CHECK:     %{{.*}}tile_1_3 = aie.tile(1, 3)
// CHECK:     %[[VAL_0:.*]] = aie.buffer(%{{.*}}tile_1_3) {sym_name = "of1_buff_0"} : memref<16xi32> 
// CHECK:     %[[VAL_1:.*]] = aie.lock(%{{.*}}tile_1_3, 0) {init = 1 : i32, sym_name = "of1_prod_lock_0"}
// CHECK:     %[[VAL_2:.*]] = aie.lock(%{{.*}}tile_1_3, 1) {init = 0 : i32, sym_name = "of1_cons_lock_0"}
// CHECK:     %[[VAL_3:.*]] = aie.buffer(%{{.*}}tile_1_2) {sym_name = "of0_buff_0"} : memref<16xi32> 
// CHECK:     %[[VAL_4:.*]] = aie.lock(%{{.*}}tile_1_2, 0) {init = 1 : i32, sym_name = "of0_prod_lock_0"}
// CHECK:     %[[VAL_5:.*]] = aie.lock(%{{.*}}tile_1_2, 1) {init = 0 : i32, sym_name = "of0_cons_lock_0"}
// CHECK:   }
// CHECK: }

module @viaSharedMem {
 aie.device(npu1) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)

    aie.objectfifo @of0 (%tile12, {%tile13}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of1 (%tile12, {%tile13}, 1 : i32) {via_shared_mem = 1 : i32} : !aie.objectfifo<memref<16xi32>>
 }
}
