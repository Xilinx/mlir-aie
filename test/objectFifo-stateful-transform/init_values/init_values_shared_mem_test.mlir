//===- init_values_shared_mem_test.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_1:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of0_buff_0"} : memref<2x2xi32> = dense<{{\[\[}}0, 1], [2, 3]]>
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of0_buff_1"} : memref<2x2xi32> = dense<{{\[\[}}4, 5], [6, 7]]>
// CHECK:           %[[VAL_3:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "of0_prod_lock_0"}
// CHECK:           %[[VAL_4:.*]] = aie.lock(%[[VAL_0]]) {init = 2 : i32, sym_name = "of0_cons_lock_0"}
// CHECK:           %[[VAL_5:.*]] = aie.tile(1, 3)
// CHECK:           %[[VAL_6:.*]] = aie.buffer(%[[VAL_5]]) {sym_name = "of1_buff_0"} : memref<2x2xi32> = dense<{{\[\[}}0, 1], [2, 3]]>
// CHECK:           %[[VAL_7:.*]] = aie.buffer(%[[VAL_5]]) {sym_name = "of1_buff_1"} : memref<2x2xi32> = dense<{{\[\[}}4, 5], [6, 7]]>
// CHECK:           %[[VAL_8:.*]] = aie.lock(%[[VAL_5]]) {init = 0 : i32, sym_name = "of1_prod_lock_0"}
// CHECK:           %[[VAL_9:.*]] = aie.lock(%[[VAL_5]]) {init = 2 : i32, sym_name = "of1_cons_lock_0"}
// CHECK:         }

module @init_shared_mem {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)

    aie.objectfifo @of0 (%tile12, {%tile13}, 2 : i32) : !aie.objectfifo<memref<2x2xi32>> = [dense<[[0, 1], [2, 3]]> : memref<2x2xi32>,
                                                                                            dense<[[4, 5], [6, 7]]> : memref<2x2xi32>]

    aie.objectfifo @of1 (%tile12, {%tile13}, 2 : i32) : !aie.objectfifo<memref<2x2xi32>> = [dense<[[0, 1], [2, 3]]> : memref<2x2xi32>,
                                                                                            dense<[[4, 5], [6, 7]]> : memref<2x2xi32>]
    aie.objectfifo.allocate @of1 (%tile13)
 }
}
