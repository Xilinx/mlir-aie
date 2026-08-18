//===- allocate_test.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_1:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of2_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_2:.*]] = aie.lock(%[[VAL_0]]) {init = 1 : i32, sym_name = "of2_prod_lock_0"}
// CHECK:           %[[VAL_3:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "of2_cons_lock_0"}
// CHECK:           %[[VAL_4:.*]] = aie.tile(1, 3)
// CHECK:           %[[VAL_5:.*]] = aie.buffer(%[[VAL_4]]) {sym_name = "of0_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_4]]) {init = 1 : i32, sym_name = "of0_prod_lock_0"}
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_4]]) {init = 0 : i32, sym_name = "of0_cons_lock_0"}
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_4]]) {sym_name = "of1_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_9:.*]] = aie.lock(%[[VAL_4]]) {init = 1 : i32, sym_name = "of1_prod_lock_0"}
// CHECK:           %[[VAL_10:.*]] = aie.lock(%[[VAL_4]]) {init = 0 : i32, sym_name = "of1_cons_lock_0"}
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
