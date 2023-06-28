//===- allocation_info_test.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
// Date: May 20th 2023
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK: module @alloc {
// CHECK:   AIE.device(xcve2302) {
// CHECK:     %0 = AIE.tile(2, 0)
// CHECK:     %1 = AIE.tile(2, 2)
// CHECK:     %2 = AIE.tile(2, 3)
// CHECK:     AIE.flow({{.*}}, DMA : {{.*}}, {{.*}}, DMA : {{.*}})
// CHECK:     AIE.flow({{.*}}, DMA : {{.*}}, {{.*}}, DMA : {{.*}})
// CHECK:     AIE.flow({{.*}}, DMA : {{.*}}, {{.*}}, DMA : {{.*}})
// CHECK:     AIE.flow({{.*}}, DMA : {{.*}}, {{.*}}, DMA : {{.*}})
// CHECK:   }
// CHECK:   AIE.shimDMAAllocation({{.*}}, {{.*}}, {{.*}}, 2)
// CHECK:   AIE.shimDMAAllocation({{.*}}, {{.*}}, {{.*}}, 2)
// CHECK:   AIE.shimDMAAllocation({{.*}}, {{.*}}, {{.*}}, 2)
// CHECK:   AIE.shimDMAAllocation({{.*}}, {{.*}}, {{.*}}, 2)
// CHECK: }

module @alloc {
    AIE.device(xcve2302) {
        %tile20 = AIE.tile(2, 0)
        %tile22 = AIE.tile(2, 2)
        %tile23 = AIE.tile(2, 3)

        %connect1 = AIE.objectFifo.createObjectFifo(%tile20, {%tile22}, 2 : i32) {sym_name = "of_in_0"} : !AIE.objectFifo<memref<64xi16>>
        %connect2 = AIE.objectFifo.createObjectFifo(%tile22, {%tile20}, 2 : i32) {sym_name = "of_out_0"} : !AIE.objectFifo<memref<64xi16>>

        %connect3 = AIE.objectFifo.createObjectFifo(%tile20, {%tile23}, 2 : i32) {sym_name = "of_in_1"} : !AIE.objectFifo<memref<64xi16>>
        %connect4 = AIE.objectFifo.createObjectFifo(%tile23, {%tile20}, 2 : i32) {sym_name = "of_out_1"} : !AIE.objectFifo<memref<64xi16>>
    }
}