//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-unroll-runtime-sequence-loops \
// RUN:         --aie-assign-runtime-sequence-bd-ids \
// RUN:         --verify-diagnostics --split-input-file %s

// A rolled ping-pong over a RUNTIME-bound loop (upper bound %n is a block
// argument). Unrolling cannot expand it, so the scf.for survives to allocation.
// The static path cannot lower runtime-valued control flow (the runtime
// sequence becomes a flat, branchless NPU instruction stream), so it is
// rejected with a pointer to the dynamic path. (A constant-trip version of this
// exact loop unrolls and allocates fine -- see
// unroll-runtime-sequence-loops/good-pingpong.mlir.)

aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @runtime_bound_pingpong(%arg0: memref<1024xi32>,
                                               %n: index) {
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %init = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = %c0_i32 len = %c256_i32 sizes = [] strides = [])
      aie.end
    }
    aiex.dma_start_task(%init)
    // expected-error@+1 {{Runtime-valued control flow in a runtime sequence is not supported}}
    %last = scf.for %i = %c1 to %n step %c1 iter_args(%prev = %init) -> (index) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = %c0_i32 len = %c256_i32 sizes = [] strides = [])
        aie.end
      }
      aiex.dma_start_task(%t)
      aiex.dma_free_task(%prev)
      scf.yield %t : index
    }
    aiex.dma_await_task(%last)
    aiex.dma_free_task(%last)
  }
}
