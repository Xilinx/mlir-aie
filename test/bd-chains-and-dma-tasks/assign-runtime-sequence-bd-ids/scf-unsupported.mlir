//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids --verify-diagnostics --split-input-file %s

// The static allocator runs on straight-line IR: constant-trip scf.for is
// unrolled and constant-predicate scf.if is folded before this pass. Any scf op
// that survives is runtime-valued and cannot be lowered to the flat, branchless
// NPU instruction stream, so it is rejected with a pointer to the dynamic path
// (rather than miscompiled or crashing, as the program-order allocator did).

// Runtime-predicate scf.if (predicate is a block argument, so canonicalize
// cannot fold it).
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence(%arg0: memref<8xi16>, %c: i1) {
    // expected-error@+1 {{Runtime-valued control flow in a runtime sequence is not supported}}
    scf.if %c {
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16> offset = %c0_i32 len = %c8_i32 sizes = [] strides = [])
        aie.end
      }
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}



// -----

// Runtime-bound scf.for (upper bound is a block argument).
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence(%arg0: memref<8xi16>, %n: index) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // expected-error@+1 {{Runtime-valued control flow in a runtime sequence is not supported}}
    scf.for %i = %c0 to %n step %c1 {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16> offset = %c0_i32 len = %c8_i32 sizes = [] strides = [])
        aie.end
      }
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}



// -----

// scf.while is never lowered by the runtime-sequence path at all.
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence(%arg0: memref<8xi16>, %c: i1) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    // expected-error@+1 {{Runtime-valued control flow in a runtime sequence is not supported}}
    scf.while : () -> () {
      scf.condition(%c)
    } do {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16> offset = %c0_i32 len = %c8_i32 sizes = [] strides = [])
        aie.end
      }
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
      scf.yield
    }
  }
}



// -----

// A free whose task value is not a dma_configure_task result. Rejected cleanly
// rather than crashing the recycle path.
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence(%arg0: memref<8xi16>, %other: index) {
    // expected-error@+1 {{does not reference a valid configure_task operation}}
    aiex.dma_free_task(%other)
  }
}
