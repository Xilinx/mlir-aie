//===- full_stream_AIE2.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s

// CHECK: module @full_stream_AIE2 {
// CHECK:   aie.device(xcve2302) {
// CHECK:     %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK:     %[[VAL_2:.*]] = aie.tile(3, 3)
// CHECK:     aie.flow(%tile_1_2, Core : 0, %tile_3_3, Core : 0)
// CHECK:   }
// CHECK: }

module @full_stream_AIE2 {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @of_full_stream (%tile12, {%tile33}, 2 : i32) {prod_port = #aie.end_port<Core : 0>, cons_ports = [#aie.end_port<Core : 0>]} : !aie.objectfifo<memref<16xi32>>
 }
}
