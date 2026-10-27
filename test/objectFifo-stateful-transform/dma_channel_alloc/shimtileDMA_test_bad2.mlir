//===- shimtileDMA_test_bad2.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" %s 2>&1 | FileCheck %s

// CHECK:   error: 'aie.tile' op number of input DMA channel exceeded!

module @shimtileDMA_channels {
    aie.device(xcvc1902) {
        %tile20 = aie.tile(2, 0)
        %tile33 = aie.tile(3, 3)

        aie.external_buffer @buff0 : memref<16xi32>
        aie.lock @lock0(%tile20, 0)
        aie.external_buffer @buff1 : memref<16xi32>
        aie.lock @lock1(%tile20, 1)
        aie.external_buffer @buff2 : memref<16xi32>
        aie.lock @lock2(%tile20, 2)

        aie.objectfifo @objfifo (%tile33, {%tile20}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

        %mem12 = aie.shim_dma(%tile20) {
            %dma1 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
        ^bb1:
            %c1_ul1 = arith.constant 1 : i32
            aie.use_lock(@lock0, Acquire, %c1_ul1)
            %buff0 = aie.get_buffer @buff0 : memref<16xi32>
            aie.dma_bd(%buff0 : memref<16xi32> offset = 0 len = 16)
            %c0_ul2 = arith.constant 0 : i32
            aie.use_lock(@lock0, Release, %c0_ul2)
            aie.next_bd ^bb2
        ^bb2:
            %c1_ul3 = arith.constant 1 : i32
            aie.use_lock(@lock1, Acquire, %c1_ul3)
            %buff1 = aie.get_buffer @buff1 : memref<16xi32>
            aie.dma_bd(%buff1 : memref<16xi32> offset = 0 len = 16)
            %c0_ul4 = arith.constant 0 : i32
            aie.use_lock(@lock1, Release, %c0_ul4)
            aie.next_bd ^bb1
        ^bb3:
            %dma2 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
        ^bb4:
            %c0_ul5 = arith.constant 0 : i32
            aie.use_lock(@lock2, Acquire, %c0_ul5)
            %buff2 = aie.get_buffer @buff2 : memref<16xi32>
            aie.dma_bd(%buff2 : memref<16xi32> offset = 0 len = 16)
            %c1_ul6 = arith.constant 1 : i32
            aie.use_lock(@lock2, Release, %c1_ul6)
            aie.next_bd ^bb4
        ^bb5:
            aie.end
        }
    }
}
