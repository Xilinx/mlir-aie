//===- memtileDMA_test_bad2.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" %s 2>&1 | FileCheck %s

// CHECK:   error: 'aie.tile' op number of output DMA channel exceeded!
// CHECK-SAME: requires at least 7 MM2S channels, but capacity is 6
// CHECK: note: DMA endpoint @objfifo_prod_dma for ObjectFifo @objfifo

module @memtileDMA_channels {
    aie.device(xcve2302) {
        %tile11 = aie.tile(1, 1)
        %tile33 = aie.tile(3, 3)

        aie.buffer @buff0(%tile11) : memref<16xi32>
        aie.buffer @buff1(%tile11) : memref<16xi32>
        aie.buffer @buff2(%tile11) : memref<16xi32>
        aie.buffer @buff3(%tile11) : memref<16xi32>
        aie.buffer @buff4(%tile11) : memref<16xi32>
        aie.buffer @buff5(%tile11) : memref<16xi32>

        aie.objectfifo @objfifo (%tile11, {%tile33}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

        %mem11 = aie.memtile_dma(%tile11) {
            %dma1 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
        ^bb1:
            %buff0 = aie.get_buffer @buff0 : memref<16xi32>
            aie.dma_bd(%buff0 : memref<16xi32> offset = 0 len = 16)
            aie.next_bd ^bb1
        ^bb2:
            %dma2 = aie.dma_start(MM2S, 1, ^bb3, ^bb4)
        ^bb3:
            %buff2 = aie.get_buffer @buff2 : memref<16xi32>
            aie.dma_bd(%buff2 : memref<16xi32> offset = 0 len = 16)
            aie.next_bd ^bb3
        ^bb4:
            %dma3 = aie.dma_start(MM2S, 2, ^bb5, ^bb6)
        ^bb5:
            %buff2_1 = aie.get_buffer @buff2 : memref<16xi32>
            aie.dma_bd(%buff2_1 : memref<16xi32> offset = 0 len = 16)
            aie.next_bd ^bb5
        ^bb6:
            %dma4 = aie.dma_start(MM2S, 3, ^bb7, ^bb8)
        ^bb7:
            %buff2_2 = aie.get_buffer @buff2 : memref<16xi32>
            aie.dma_bd(%buff2_2 : memref<16xi32> offset = 0 len = 16)
            aie.next_bd ^bb7
        ^bb8:
            %dma5 = aie.dma_start(MM2S, 4, ^bb9, ^bb10)
        ^bb9:
            %buff2_3 = aie.get_buffer @buff2 : memref<16xi32>
            aie.dma_bd(%buff2_3 : memref<16xi32> offset = 0 len = 16)
            aie.next_bd ^bb9
        ^bb10:
            %dma6 = aie.dma_start(MM2S, 5, ^bb11, ^bb12)
        ^bb11:
            %buff2_4 = aie.get_buffer @buff2 : memref<16xi32>
            aie.dma_bd(%buff2_4 : memref<16xi32> offset = 0 len = 16)
            aie.next_bd ^bb11
        ^bb12:
            aie.end
        }
    }
}
