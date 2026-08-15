// RUN: aie-opt --aie-objectfifo-lower-dmas %s | FileCheck %s

// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// An endpoint responsible for several segments walks the objects in turn and,
// within each, the slices it owns -- each under its own locks.

module @segments {
  aie.device(xcve2302) {
    %memtile = aie.tile(2, 1)

    %b0 = aie.buffer(%memtile) {sym_name = "b0"} : memref<48xi32>
    %b1 = aie.buffer(%memtile) {sym_name = "b1"} : memref<48xi32>
    %f0 = aie.lock(%memtile) {init = 2 : i32, sym_name = "f0"}
    %u0 = aie.lock(%memtile) {init = 0 : i32, sym_name = "u0"}
    %f1 = aie.lock(%memtile) {init = 2 : i32, sym_name = "f1"}
    %u1 = aie.lock(%memtile) {init = 0 : i32, sym_name = "u1"}

    aie.objectfifo.pool @pool(%memtile) {
      depth = 2 : i32, buffers = [@b0, @b1],
      segments = [#aie.objectfifo_segment<offset = 0, size = 16,
                                          produceLock = @f0, consumeLock = @u0>,
                  #aie.objectfifo_segment<offset = 16, size = 32,
                                          produceLock = @f1, consumeLock = @u1>]
    } : memref<48xi32>

    // The gathering end fills one slice of every object.
    aie.objectfifo.dma_endpoint @part(%memtile) fills @pool {
      channelIndex = 0 : i32, segments = array<i32: 1>
    }

    // The forwarding end drains all of them.
    aie.objectfifo.dma_endpoint @whole(%memtile) drains @pool {
      channelIndex = 0 : i32
    }
  }
}

// CHECK-LABEL: @segments
// CHECK:   aie.memtile_dma(%{{.*}}) {
// CHECK:     aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:   ^bb1:
// CHECK:     aie.use_lock(%f1, AcquireGreaterEqual, %{{.*}})
// CHECK:     aie.dma_bd(%b0 : memref<48xi32> offset = 16 len = 32)
// CHECK:     aie.use_lock(%u1, Release, %{{.*}})
// CHECK:   ^bb2:
// CHECK:     aie.dma_bd(%b1 : memref<48xi32> offset = 16 len = 32)

// CHECK:     aie.dma_start(MM2S, 0, ^bb4, ^bb8)
// CHECK:   ^bb4:
// CHECK:     aie.use_lock(%u0, AcquireGreaterEqual, %{{.*}})
// CHECK:     aie.dma_bd(%b0 : memref<48xi32> offset = 0 len = 16)
// CHECK:     aie.use_lock(%f0, Release, %{{.*}})
// CHECK:   ^bb5:
// CHECK:     aie.use_lock(%u1, AcquireGreaterEqual, %{{.*}})
// CHECK:     aie.dma_bd(%b0 : memref<48xi32> offset = 16 len = 32)
// CHECK:     aie.use_lock(%f1, Release, %{{.*}})
// CHECK:   ^bb6:
// CHECK:     aie.dma_bd(%b1 : memref<48xi32> offset = 0 len = 16)
// CHECK:   ^bb7:
// CHECK:     aie.dma_bd(%b1 : memref<48xi32> offset = 16 len = 32)
