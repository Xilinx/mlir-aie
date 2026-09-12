//===- iter_count_with_consumer_loop.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll --split-input-file %s | FileCheck %s

// `iter_count` bounds the MemTile's BD chain while the consumer core runs its
// own loop. The two counts are independent, and the second case pins the
// iter_count = 1 edge, where the chain ends after a single pass.

// Both ends of the fifo end their chain after iter_count passes, which the
// hardware counts as repeat_count = iter_count - 1.
// CHECK-LABEL: @iterCountWithLoop
// CHECK:         aie.memtile_dma(%{{.*}}mem_tile_0_1)
// CHECK:           aie.dma_start(MM2S, 0, ^bb1, ^bb4, repeat_count = 4)
// CHECK:         aie.mem(%{{.*}}tile_0_2)
// CHECK:           aie.dma_start(S2MM, 0, ^bb1, ^bb4, repeat_count = 4)

// A single pass needs no repeat at all.
// CHECK-LABEL: @iterCountOne
// CHECK:         aie.memtile_dma(%{{.*}}mem_tile_0_1)
// CHECK:           aie.dma_start(MM2S, 0, ^bb1, ^bb4)
// CHECK-NOT:       repeat_count
// CHECK:         aie.mem(%{{.*}}tile_0_2)
// CHECK:           aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK-NOT:       repeat_count

module @iterCountWithLoop {
 aie.device(npu1_1col) {
    %tile01 = aie.tile(0, 1)
    %tile02 = aie.tile(0, 2)

    aie.objectfifo @of (%tile01, {%tile02}, 2 : i32) {iter_count = 5 : i32} : !aie.objectfifo<memref<16xi32>>

    %core02 = aie.core(%tile02) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c5 = arith.constant 5 : index
      scf.for %i = %c0 to %c5 step %c1 {
        %e = aie.objectfifo.acquire @of (Consume, 1) : memref<16xi32>
        %v = memref.load %e[%c0] : memref<16xi32>
        aie.objectfifo.release @of (Consume, 1)
      }
      aie.end
    }
 }
}

// -----

module @iterCountOne {
 aie.device(npu1_1col) {
    %tile01 = aie.tile(0, 1)
    %tile02 = aie.tile(0, 2)

    aie.objectfifo @of (%tile01, {%tile02}, 2 : i32) {iter_count = 1 : i32} : !aie.objectfifo<memref<16xi32>>

    %core02 = aie.core(%tile02) {
      %c0 = arith.constant 0 : index
      %e = aie.objectfifo.acquire @of (Consume, 1) : memref<16xi32>
      %v = memref.load %e[%c0] : memref<16xi32>
      aie.objectfifo.release @of (Consume, 1)
      aie.end
    }
 }
}
