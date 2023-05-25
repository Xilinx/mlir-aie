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
// CHECK:   AIE.device(ipu) {
// CHECK:     %0 = AIE.tile(0, 0)
// CHECK:     %1 = AIE.tile(0, 2)
// CHECK:     %2 = AIE.tile(0, 3)
// CHECK:     AIE.flow({{.*}}, DMA : {{.*}}, {{.*}}, DMA : {{.*}})
// CHECK:     AIE.flow({{.*}}, DMA : {{.*}}, {{.*}}, DMA : {{.*}})
// CHECK:     AIE.flow({{.*}}, DMA : {{.*}}, {{.*}}, DMA : {{.*}})
// CHECK:     AIE.flow({{.*}}, DMA : {{.*}}, {{.*}}, DMA : {{.*}})
// CHECK:   }
// CHECK:   AIE.shimDMAAllocationInfo({{.*}}, {{.*}}, {{.*}}, 0)
// CHECK:   AIE.shimDMAAllocationInfo({{.*}}, {{.*}}, {{.*}}, 0)
// CHECK:   AIE.shimDMAAllocationInfo({{.*}}, {{.*}}, {{.*}}, 0)
// CHECK:   AIE.shimDMAAllocationInfo({{.*}}, {{.*}}, {{.*}}, 0)
// CHECK: }

module @alloc {
    AIE.device(xcve2302) {
        %tile00 = AIE.tile(0, 0)
        %tile02 = AIE.tile(0, 2)
        %tile03 = AIE.tile(0, 3)

        %connect1 = AIE.objectFifo.createObjectFifo(%tile00, {%tile02}, 2) {sym_name = "of_in_0"} : !AIE.objectFifo<memref<64xi16>>
        %connect2 = AIE.objectFifo.createObjectFifo(%tile02, {%tile00}, 2) {sym_name = "of_out_0"} : !AIE.objectFifo<memref<64xi16>>

        %connect3 = AIE.objectFifo.createObjectFifo(%tile00, {%tile03}, 2) {sym_name = "of_in_1"} : !AIE.objectFifo<memref<64xi16>>
        %connect4 = AIE.objectFifo.createObjectFifo(%tile03, {%tile00}, 2) {sym_name = "of_out_1"} : !AIE.objectFifo<memref<64xi16>>
    }
}