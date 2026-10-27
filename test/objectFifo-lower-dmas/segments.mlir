// RUN: aie-opt --aie-objectfifo-lower-dmas %s | FileCheck %s

// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// An endpoint responsible for several segments walks the objects in turn and,
// within each, the slices it owns -- each under its own locks.

module @segments {
  aie.device(xcve2302) {
    %memtile = aie.tile(2, 1)

    aie.buffer @b0(%memtile) : memref<48xi32>
    aie.buffer @b1(%memtile) : memref<48xi32>
    aie.lock @f0(%memtile) {init = 2 : i32}
    aie.lock @u0(%memtile) {init = 0 : i32}
    aie.lock @f1(%memtile) {init = 2 : i32}
    aie.lock @u1(%memtile) {init = 0 : i32}

    aie.objectfifo.pool @pool(%memtile) {
      depth = 2 : i32, buffers = [@b0, @b1]
    } : memref<48xi32> {
      aie.objectfifo.segment @s0 {consumeLock = @u0, offset = 0 : i32, produceLock = @f0, size = 16 : i32}
      aie.objectfifo.segment @s1 {consumeLock = @u1, offset = 16 : i32, produceLock = @f1, size = 32 : i32}
    }

    // The gathering end fills one slice of every object.
    aie.objectfifo.dma_endpoint @part(%memtile) fills @pool {
      channelIndex = 0 : i32, segments = [@s1]
    }

    // The forwarding end drains all of them.
    aie.objectfifo.dma_endpoint @whole(%memtile) drains @pool {
      channelIndex = 0 : i32, segments = [@s0, @s1]
    }
  }
}

// CHECK-LABEL: @segments
// CHECK:   aie.memtile_dma(%{{.*}}) {
// CHECK:     aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:   ^bb1:
// CHECK:     aie.use_lock(@f1, AcquireGreaterEqual, %{{.*}})
// CHECK:     %[[PART0:.*]] = aie.get_buffer @b0 : memref<48xi32>
// CHECK:     aie.dma_bd(%[[PART0]] : memref<48xi32> offset = 16 len = 32)
// CHECK:     aie.use_lock(@u1, Release, %{{.*}})
// CHECK:   ^bb2:
// CHECK:     %[[PART1:.*]] = aie.get_buffer @b1 : memref<48xi32>
// CHECK:     aie.dma_bd(%[[PART1]] : memref<48xi32> offset = 16 len = 32)

// CHECK:     aie.dma_start(MM2S, 0, ^bb4, ^bb8)
// CHECK:   ^bb4:
// CHECK:     aie.use_lock(@u0, AcquireGreaterEqual, %{{.*}})
// CHECK:     %[[WHOLE0S0:.*]] = aie.get_buffer @b0 : memref<48xi32>
// CHECK:     aie.dma_bd(%[[WHOLE0S0]] : memref<48xi32> offset = 0 len = 16)
// CHECK:     aie.use_lock(@f0, Release, %{{.*}})
// CHECK:   ^bb5:
// CHECK:     aie.use_lock(@u1, AcquireGreaterEqual, %{{.*}})
// CHECK:     %[[WHOLE0S1:.*]] = aie.get_buffer @b0 : memref<48xi32>
// CHECK:     aie.dma_bd(%[[WHOLE0S1]] : memref<48xi32> offset = 16 len = 32)
// CHECK:     aie.use_lock(@f1, Release, %{{.*}})
// CHECK:   ^bb6:
// CHECK:     %[[WHOLE1S0:.*]] = aie.get_buffer @b1 : memref<48xi32>
// CHECK:     aie.dma_bd(%[[WHOLE1S0]] : memref<48xi32> offset = 0 len = 16)
// CHECK:   ^bb7:
// CHECK:     %[[WHOLE1S1:.*]] = aie.get_buffer @b1 : memref<48xi32>
// CHECK:     aie.dma_bd(%[[WHOLE1S1]] : memref<48xi32> offset = 16 len = 32)
