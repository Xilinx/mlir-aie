//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids --verify-diagnostics --split-input-file %s

// Control-flow forms that genuinely cannot be statically allocated are rejected
// with a clear diagnostic rather than miscompiled (or crashing, as the
// program-order allocator did). Constant-trip rolled ping-pong is handled by
// unrolling before this pass (see unroll-runtime-sequence-loops/good-pingpong.mlir)
// and scf.if value joins are supported (see good-if-join.mlir); the cases below
// remain rejections because no correct static allocation exists for them.

// Task configured in a loop with no reachable completion sync (leaks one BD per
// iteration).
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence(%arg0: memref<8xi16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %c4 step %c1 {
      // expected-error@+1 {{configured in a loop is never completed}}
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// Handle carried UNCHANGED across a loop back-edge (yield the iter_arg itself).
// This is a def-use cycle; the analysis must terminate and reject, not hang.
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence(%arg0: memref<8xi16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    // expected-error@+1 {{cannot be statically resolved to a single completion}}
    %init = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
      aie.end
    }
    aiex.dma_start_task(%init)
    %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%p = %init) -> (index) {
      scf.yield %p : index
    }
    aiex.dma_free_task(%r)
  }
}

// -----

// Handle both freed and re-yielded across the back-edge in the same iteration:
// two continuations, cannot be linearized => reject (must not crash).
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence(%arg0: memref<8xi16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    // expected-error@+1 {{cannot be statically resolved to a single completion}}
    %init = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
      aie.end
    }
    aiex.dma_start_task(%init)
    %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%p = %init) -> (index) {
      aiex.dma_free_task(%p)
      scf.yield %p : index
    }
    aiex.dma_free_task(%r)
  }
}

// -----

// Free of a value that is not a dma_configure_task result (here, an scf.for
// result). Must be rejected cleanly, not crash the recycle path.
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence(%arg0: memref<8xi16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%p = %c0) -> (index) {
      scf.yield %p : index
    }
    // expected-error@+1 {{does not reference a valid configure_task operation}}
    aiex.dma_free_task(%r)
  }
}
