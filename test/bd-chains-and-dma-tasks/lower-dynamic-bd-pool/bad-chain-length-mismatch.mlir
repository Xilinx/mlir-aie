//===- bad-chain-length-mismatch.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-lower-dynamic-bd-pool --verify-diagnostics %s

// A task position carries one id per BD of its chain, so the two branches of
// an scf.if must yield chains of the same length.

aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @if_mismatch(%arg0: memref<1024xi32>, %cond: i1) {
    // expected-error@+1 {{yields from its branches a 2-BD chain and a 1-BD chain at the same position}}
    %r = scf.if %cond -> (index) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 128)
        aie.next_bd ^bd1
      ^bd1:
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 128 len = 128)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
      scf.yield %t : index
    } else {
      %t2 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 512 len = 256)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t2)
      aiex.dma_await_task(%t2)
      scf.yield %t2 : index
    }
    aiex.dma_free_task(%r)
  }
}

// -----

// Likewise an scf.for's init and per-iteration yield.

aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @for_mismatch(%arg0: memref<1024xi32>, %n: index) {
    %c1 = arith.constant 1 : index
    %init = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
      aie.end
    }
    aiex.dma_start_task(%init)
    // expected-error@+1 {{carries from its init and its yield a 1-BD chain and a 2-BD chain at the same position}}
    %last = scf.for %i = %c1 to %n step %c1 iter_args(%prev = %init) -> (index) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 128)
        aie.next_bd ^bd1
      ^bd1:
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 128 len = 128)
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
