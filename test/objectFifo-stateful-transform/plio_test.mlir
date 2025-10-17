//===- plio_test.mlir -------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK:  aie.flow(%{{.*}}tile_2_0, PLIO : 0, %{{.*}}tile_2_2, DMA : 0)
// CHECK:  aie.flow(%{{.*}}tile_2_2, DMA : 0, %{{.*}}tile_2_0, PLIO : 0)
// CHECK:  aie.flow(%{{.*}}tile_2_2, DMA : 1, %{{.*}}tile_2_3, DMA : 0)
// CHECK:  aie.flow(%{{.*}}tile_2_2, DMA : 1, %{{.*}}tile_2_0, PLIO : 1)
// CHECK:  aie.shim_dma_allocation @of_0_shim_alloc(MM2S, 0, 2) {plio = true}
// CHECK:  aie.shim_dma_allocation @of_1_shim_alloc(S2MM, 0, 2) {plio = true}
// CHECK:  aie.shim_dma_allocation @of_2_shim_alloc(S2MM, 1, 2) {plio = true}

module @plio {
    aie.device(xcve2302) {
        %tile20 = aie.tile(2, 0)
        %tile22 = aie.tile(2, 2)
        %tile23 = aie.tile(2, 3)

        aie.objectfifo @of_0 (%tile20, {%tile22}, 2 : i32) {plio = true} : !aie.objectfifo<memref<64xi16>>
        aie.objectfifo @of_1 (%tile22, {%tile20}, 2 : i32) {plio = true} : !aie.objectfifo<memref<64xi16>>
        aie.objectfifo @of_2 (%tile22, {%tile20, %tile23}, 2 : i32) {plio = true} : !aie.objectfifo<memref<64xi16>>
    }
}
