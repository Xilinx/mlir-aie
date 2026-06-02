//===- hw_repeat_optimization_test.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --split-input-file %s | FileCheck %s

// Test: Hardware repeat optimization for repeat_count with single buffer (depth=1).
// When numBlocks == 1 and repeat_count > 1, the stateful transform generates
// 1 BD with dma_start repeat_count instead of N identical BDs.

// MemTile with repeat_count=3, depth=1: 1 BD + repeat_count=2
// CHECK-LABEL: @memtile_hw_repeat
// CHECK:       %memtile_dma
// CHECK:         aie.dma_start(MM2S, 0, ^[[BD:.*]], ^{{.*}}, repeat_count = 2)
// CHECK:       ^[[BD]]:
// CHECK:         aie.dma_bd(
// CHECK:         aie.next_bd ^[[BD]]
module @memtile_hw_repeat {
  aie.device(npu2) {
    %tile11 = aie.tile(1, 1)
    %tile12 = aie.tile(1, 2)

    aie.objectfifo @of1(%tile11, {%tile12}, 1 : i32) {repeat_count = 3 : i32} : !aie.objectfifo<memref<16xi32>>
  }
}

// -----

// Core tile with repeat_count=4, depth=1: 1 BD + repeat_count=3
// CHECK-LABEL: @core_hw_repeat
// CHECK:       %mem_0_2 = aie.mem
// CHECK:         aie.dma_start(MM2S, 0, ^[[BD:.*]], ^{{.*}}, repeat_count = 3)
// CHECK:       ^[[BD]]:
// CHECK:         aie.dma_bd(
// CHECK:         aie.next_bd ^[[BD]]
module @core_hw_repeat {
  aie.device(npu2) {
    %tile02 = aie.tile(0, 2)
    %tile03 = aie.tile(0, 3)

    aie.objectfifo @of1(%tile02, {%tile03}, 1 : i32) {repeat_count = 4 : i32} : !aie.objectfifo<memref<64xi32>>
  }
}

// -----

// depth=2 with repeat_count=3: should NOT use hardware repeat (multiple buffers)
// CHECK-LABEL: @no_hw_repeat_depth2
// CHECK:       %mem_0_2 = aie.mem
// CHECK:         aie.dma_start(MM2S, 0, ^{{.*}}, ^{{.*}})
// CHECK-NOT:     repeat_count
module @no_hw_repeat_depth2 {
  aie.device(npu2) {
    %tile02 = aie.tile(0, 2)
    %tile03 = aie.tile(0, 3)

    aie.objectfifo @of1(%tile02, {%tile03}, 2 : i32) {repeat_count = 3 : i32} : !aie.objectfifo<memref<64xi32>>
  }
}

// -----

// depth=1 with repeat_count=1: no repeat_count in output (trivial case)
// CHECK-LABEL: @no_hw_repeat_count1
// CHECK:       %memtile_dma
// CHECK:         aie.dma_start(MM2S, 0, ^[[BD:.*]], ^{{.*}})
// CHECK-NOT:     repeat_count
// CHECK:       ^[[BD]]:
// CHECK:         aie.dma_bd(
// CHECK:         aie.next_bd ^[[BD]]
module @no_hw_repeat_count1 {
  aie.device(npu2) {
    %tile11 = aie.tile(1, 1)
    %tile12 = aie.tile(1, 2)

    aie.objectfifo @of1(%tile11, {%tile12}, 1 : i32) {repeat_count = 1 : i32} : !aie.objectfifo<memref<16xi32>>
  }
}

// -----

// depth=1, repeat_count=3 with distribute: should NOT use hardware repeat
// (joinDistribFactor > 1 means BDs interleave across sub-buffers)
// CHECK-LABEL: @no_hw_repeat_distribute
// CHECK:       %memtile_dma
// The S2MM (receive) side of the linked objectfifo on the MemTile
// should NOT have repeat_count because distribute interleaves BDs.
// CHECK:         aie.dma_start(S2MM, 0, ^{{.*}}, ^{{.*}})
// CHECK-NOT:     repeat_count
module @no_hw_repeat_distribute {
  aie.device(npu2) {
    %tile11 = aie.tile(1, 1)
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    %tile21 = aie.tile(2, 1)

    aie.objectfifo @of_in(%tile21, {%tile11}, 1 : i32) {repeat_count = 3 : i32} : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @of_out1(%tile11, {%tile12}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of_out2(%tile11, {%tile13}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@of_in] -> [@of_out1, @of_out2]([] [0, 16])
  }
}
