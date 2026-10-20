//===- channel-tracks.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Two parallel runs on the same line read as one, so arrows sharing a corridor
// are given tracks of their own, and arrows meeting the same box are given
// their own point on its edge. Four flows cross the same switchbox here and
// four buffers feed the same kernel; none of the eight may share a line.

// REQUIRES: aie-visualize
// RUN: aie-visualize --emit-dot --arrows=channel %s | FileCheck %s

// Every arrow carries its own geometry.
// CHECK-COUNT-8: -> {{.*}}pos="

// The four runs across the switchbox each take a different track, so the
// x each leaves on differs. Were they sharing a line these would repeat.
// CHECK-NOT: pos="[[X:[-0-9.]+]],{{[-0-9.]+}} {{.*}} pos="[[X]],

module {
  aie.device(npu1_1col) {
    func.func private @consume(memref<16xi32>, memref<16xi32>, memref<16xi32>,
                               memref<16xi32>)

    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %core = aie.tile(0, 2)

    %a = aie.buffer(%core) {sym_name = "a"} : memref<16xi32>
    %b = aie.buffer(%core) {sym_name = "b"} : memref<16xi32>
    %c = aie.buffer(%core) {sym_name = "c"} : memref<16xi32>
    %d = aie.buffer(%core) {sym_name = "d"} : memref<16xi32>

    aie.core(%core) {
      func.call @consume(%a, %b, %c, %d)
          : (memref<16xi32>, memref<16xi32>, memref<16xi32>, memref<16xi32>)
          -> ()
      aie.end
    }

    aie.mem(%core) {
      %0 = aie.dma_start(S2MM, 0, ^bd0, ^ch1)
    ^bd0:
      aie.dma_bd(%a : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^end
    ^ch1:
      %1 = aie.dma_start(S2MM, 1, ^bd1, ^end)
    ^bd1:
      aie.dma_bd(%b : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^end
    ^end:
      aie.end
    }

    // Four flows pass through the memtile's switchbox on their way up.
    aie.flow(%shim, DMA : 0, %core, DMA : 0)
      via (%shim : DMA : 0 -> North : 0,
           %mem : South : 0 -> North : 0,
           %core : South : 0 -> DMA : 0)
    aie.flow(%shim, DMA : 1, %core, DMA : 1)
      via (%shim : DMA : 1 -> North : 1,
           %mem : South : 1 -> North : 1,
           %core : South : 1 -> DMA : 1)
  }
}
