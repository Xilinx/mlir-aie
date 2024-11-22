//===- tileDMA_test_bad4.mlir ----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-objectFifo-stateful-transform %s 2>&1 | FileCheck %s

// CHECK:   error: 'aie.tile' op number of output DMA channel exceeded!

module @tileDMA_channels {
    aie.device(xcve2302) {
        %tile11 = aie.tile(1, 1)
        %tile33 = aie.tile(3, 3)

        %buff0 = aie.buffer(%tile11) : memref<16xi32>
        %buff1 = aie.buffer(%tile11) : memref<16xi32>
        %buff2 = aie.buffer(%tile11) : memref<16xi32>
        %buff3 = aie.buffer(%tile11) : memref<16xi32>
        %buff4 = aie.buffer(%tile11) : memref<16xi32>
        %buff5 = aie.buffer(%tile11) : memref<16xi32>

        aie.objectfifo @objfifo (%tile11, {%tile33}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

        %mem11 = aie.memtile_dma(%tile11) {
            %dma1 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
        ^bb1:
            aie.dma_bd(%buff0 : memref<16xi32>, 0, 16)
            aie.next_bd ^bb1
        ^bb2:
            %dma2 = aie.dma_start(MM2S, 1, ^bb3, ^bb4)
        ^bb3:
            aie.dma_bd(%buff2 : memref<16xi32>, 0, 16)
            aie.next_bd ^bb3
        ^bb4:
            %dma3 = aie.dma_start(MM2S, 2, ^bb5, ^bb6)
        ^bb5:
            aie.dma_bd(%buff2 : memref<16xi32>, 0, 16)
            aie.next_bd ^bb5
        ^bb6:
            %dma4 = aie.dma_start(MM2S, 3, ^bb7, ^bb8)
        ^bb7:
            aie.dma_bd(%buff2 : memref<16xi32>, 0, 16)
            aie.next_bd ^bb7
        ^bb8:
            %dma5 = aie.dma_start(MM2S, 4, ^bb9, ^bb10)
        ^bb9:
            aie.dma_bd(%buff2 : memref<16xi32>, 0, 16)
            aie.next_bd ^bb9
        ^bb10:
            %dma6 = aie.dma_start(MM2S, 5, ^bb11, ^bb12)
        ^bb11:
            aie.dma_bd(%buff2 : memref<16xi32>, 0, 16)
            aie.next_bd ^bb11
        ^bb12:
            aie.end
        }
    }
}
