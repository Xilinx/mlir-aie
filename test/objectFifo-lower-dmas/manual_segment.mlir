// RUN: aie-opt --split-input-file --aie-objectfifo-allocate --aie-objectfifo-lower-dmas --aie-objectfifo-lower-cores %s | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// `manualFill` and `manualDrain` sit on a segment, not on the pool, so one
// pool may hand one segment to a DMA program the author writes and leave the
// rest to the compiler. The segment the author took gets no buffer descriptor
// chain; its sibling gets the usual one; and the core waits on both.

module @manual_fill_segment {
  aie.device(xcve2302) {
    %tile = aie.tile(1, 2)
    %shim = aie.tile(1, 0)

    aie.buffer @b0(%tile) : memref<48xi32>
    aie.buffer @b1(%tile) : memref<48xi32>
    aie.lock @f0(%tile) {init = 2 : i32}
    aie.lock @u0(%tile) {init = 0 : i32}
    aie.lock @f1(%tile) {init = 2 : i32}
    aie.lock @u1(%tile) {init = 0 : i32}

    aie.objectfifo.pool @pool(%tile) {
      depth = 2 : i32, buffers = [@b0, @b1]
    } : memref<48xi32> {
      aie.objectfifo.segment @s0 {consumeLock = @u0, manualFill, offset = 0 : i32, produceLock = @f0, size = 16 : i32}
      aie.objectfifo.segment @s1 {consumeLock = @u1, offset = 16 : i32, produceLock = @f1, size = 32 : i32}
    }

    aie.objectfifo.dma_endpoint @fill(%tile) fills @pool {segments = [@s1]}
    aie.route_endpoint @src(%shim) DMA {channelIndex = 0 : i32}
    aie.route from @src to [@fill]

    aie.objectfifo.core_endpoint @reader(%tile) drains @pool {segments = [@s0, @s1]}
    %core = aie.core(%tile) {
      %e = aie.objectfifo.acquire @reader (1) : memref<48xi32>
      aie.objectfifo.release @reader (1)
      aie.end
    }
  }
}

// The generated chain covers @s1 alone: offset 16, length 32, under @s1's
// locks. Every line of the chain is pinned, so a descriptor for @s0 would have
// nowhere to hide.
// CHECK-LABEL: @manual_fill_segment
// CHECK:        aie.mem(%{{.*}}) {
// CHECK-NEXT:     aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK-NEXT:   ^bb1:
// CHECK-NEXT:     %{{.*}} = arith.constant 1 : i32
// CHECK-NEXT:     aie.use_lock(@f1, AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT:     %[[B0:.*]] = aie.get_buffer @b0 : memref<48xi32>
// CHECK-NEXT:     aie.dma_bd(%[[B0]] : memref<48xi32> offset = 16 len = 32)
// CHECK-NEXT:     %{{.*}} = arith.constant 1 : i32
// CHECK-NEXT:     aie.use_lock(@u1, Release, %{{.*}})
// CHECK-NEXT:     aie.next_bd ^bb2
// CHECK-NEXT:   ^bb2:
// CHECK-NEXT:     %{{.*}} = arith.constant 1 : i32
// CHECK-NEXT:     aie.use_lock(@f1, AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT:     %[[B1:.*]] = aie.get_buffer @b1 : memref<48xi32>
// CHECK-NEXT:     aie.dma_bd(%[[B1]] : memref<48xi32> offset = 16 len = 32)
// CHECK-NEXT:     %{{.*}} = arith.constant 1 : i32
// CHECK-NEXT:     aie.use_lock(@u1, Release, %{{.*}})
// CHECK-NEXT:     aie.next_bd ^bb1
// CHECK-NEXT:   ^bb3:
// CHECK-NEXT:     aie.end

// -----

// Left to itself, the allocator gives the manual segment locks too, and names
// them, which is how a hand-written program reaches them.

module @manual_fill_segment_allocated {
  aie.device(xcve2302) {
    %tile = aie.tile(1, 2)
    %shim = aie.tile(1, 0)

    aie.objectfifo.pool @pool(%tile) {depth = 2 : i32} : memref<48xi32> {
      aie.objectfifo.segment @s0 {manualFill, offset = 0 : i32, size = 16 : i32}
      aie.objectfifo.segment @s1 {offset = 16 : i32, size = 32 : i32}
    }

    aie.objectfifo.dma_endpoint @fill(%tile) fills @pool {segments = [@s1]}
    aie.route_endpoint @src(%shim) DMA {channelIndex = 0 : i32}
    aie.route from @src to [@fill]

    aie.objectfifo.core_endpoint @reader(%tile) drains @pool {segments = [@s0, @s1]}
    %core = aie.core(%tile) {
      %e = aie.objectfifo.acquire @reader (1) : memref<48xi32>
      aie.objectfifo.release @reader (1)
      aie.end
    }
  }
}

// CHECK-LABEL: @manual_fill_segment_allocated
// CHECK-DAG: aie.lock @[[F0:pool_prod_lock_0]](%{{.*}}) {init = 2 : i32}
// CHECK-DAG: aie.lock @[[U0:pool_cons_lock_0]](%{{.*}}) {init = 0 : i32}
// CHECK-DAG: aie.lock @[[F1:pool_prod_lock_1]](%{{.*}}) {init = 2 : i32}
// CHECK-DAG: aie.lock @[[U1:pool_cons_lock_1]](%{{.*}}) {init = 0 : i32}

// The core holds the whole object, so it waits on both segments and frees both.
// CHECK:   aie.core(%{{.*}}) {
// CHECK:     aie.use_lock(@[[U0]], AcquireGreaterEqual, %{{.*}})
// CHECK:     aie.use_lock(@[[U1]], AcquireGreaterEqual, %{{.*}})
// CHECK:     aie.use_lock(@[[F0]], Release, %{{.*}})
// CHECK:     aie.use_lock(@[[F1]], Release, %{{.*}})
