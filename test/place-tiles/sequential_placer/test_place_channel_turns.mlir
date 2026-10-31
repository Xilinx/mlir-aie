//===- test_place_channel_turns.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --split-input-file --aie-place-tiles --aie-objectFifo-stateful-transform %s 2>&1 | FileCheck %s
// RUN: not aie-opt --split-input-file --aie-place-tiles --aie-objectFifo-stateful-transform %s 2>&1 | FileCheck %s --check-prefix=ERR

// The placer counts a core's input channels the way objectFIFO allocation
// shares them: once the core runs out, the fifos one mem tile sends it, which
// the core takes in a fixed order, count once. Three fifos reach a core with
// two input channels; two of them take turns on one.
// CHECK-LABEL: module @three_fifos_two_channels
// CHECK:       aie.mem(%[[CORE:.*]])
// CHECK:       aie.dma_start(S2MM
// CHECK:       aie.dma_bd(%b__cons__buf0
// CHECK:       aie.dma_bd(%c__cons__buf0
module @three_fifos_two_channels {
  aie.device(npu2_1col) {
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %m = aie.logical_tile<MemTile>(?, ?)
    %t = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @a_in(%s, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a(%m, {%t}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@a_in] -> [@a] ([] [])
    aie.objectfifo @bc_in(%s, {%m}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @b(%m, {%t}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c(%m, {%t}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@bc_in] -> [@b, @c] ([] [0, 16])
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.for %i = %c0 to %c8 step %c1 {
        %a0, %a1 = aie.objectfifo.acquire @a(Consume, 2) : memref<16xi32>, memref<16xi32>
        %b = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
        %c = aie.objectfifo.acquire @c(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @a(Consume, 2)
        aie.objectfifo.release @b(Consume, 1)
        aie.objectfifo.release @c(Consume, 1)
      }
      aie.end
    }
  }
}

// -----

// A core that takes c only under a branch needs its objects in no fixed
// order, so no two of its fifos can take turns and it does not fit.
// ERR:         error: {{.*}}requires 3 input/0 output DMA channels, but only 2 input/2 output available
module @branch {
  aie.device(npu2_1col) {
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %m = aie.logical_tile<MemTile>(?, ?)
    %t = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @a_in(%s, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a(%m, {%t}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@a_in] -> [@a] ([] [])
    aie.objectfifo @bc_in(%s, {%m}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @b(%m, {%t}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c(%m, {%t}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@bc_in] -> [@b, @c] ([] [0, 16])
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.for %i = %c0 to %c8 step %c1 {
        %a0 = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
        %first = arith.cmpi eq, %i, %c0 : index
        scf.if %first {
          %b = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
          aie.objectfifo.release @b(Consume, 1)
        }
        %c = aie.objectfifo.acquire @c(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @a(Consume, 1)
        aie.objectfifo.release @c(Consume, 1)
      }
      aie.end
    }
  }
}
