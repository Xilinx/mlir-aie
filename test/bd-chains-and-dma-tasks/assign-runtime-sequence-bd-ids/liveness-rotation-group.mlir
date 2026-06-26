//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.

// RUN: aie-opt --aie-test-runtime-bd-liveness --verify-diagnostics --split-input-file %s

// Unit test for resolveLoopRotationGroup: per rolled-ping-pong body, the
// reported window width W = D+1, chain length C, and member count (D prologues
// + 1 body). Independent of BD-ID allocation.

// Depth D=1, single-bd chain: width 2, chain 1, 2 members.
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  // expected-remark@+1 {{bd-peak: tile(0,0) peak=2}}
  aie.runtime_sequence(%arg0: memref<8xi16>) {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    // expected-remark@+1 {{bd-liveness: backedges=0 kill=aiex.dma_free_task}}
    %init = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
      aie.end
    }
    aiex.dma_start_task(%init)
    %last = scf.for %i = %c1 to %c4 step %c1 iter_args(%prev = %init) -> (index) {
      // expected-remark@+2 {{bd-liveness: backedges=1 kill=aiex.dma_free_task in-loop}}
      // expected-remark@+1 {{bd-rotation: width=2 chain=1 members=2}}
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      aiex.dma_start_task(%t)
      aiex.dma_free_task(%prev)
      scf.yield %t : index
    }
    aiex.dma_free_task(%last)
  }
}

// -----

// Depth D=1, two-bd chain (C=2): width 2, chain 2, 2 members.
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  // expected-remark@+1 {{bd-peak: tile(0,0) peak=4}}
  aie.runtime_sequence(%arg0: memref<8xi16>) {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    // expected-remark@+1 {{bd-liveness: backedges=0 kill=aiex.dma_free_task}}
    %init = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<8xi16>, 0, 4)
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%arg0 : memref<8xi16>, 4, 4)
      aie.end
    }
    aiex.dma_start_task(%init)
    %last = scf.for %i = %c1 to %c4 step %c1 iter_args(%prev = %init) -> (index) {
      // expected-remark@+2 {{bd-liveness: backedges=1 kill=aiex.dma_free_task in-loop}}
      // expected-remark@+1 {{bd-rotation: width=2 chain=2 members=2}}
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 4)
        aie.next_bd ^bd1
      ^bd1:
        aie.dma_bd(%arg0 : memref<8xi16>, 4, 4)
        aie.end
      }
      aiex.dma_start_task(%t)
      aiex.dma_free_task(%prev)
      scf.yield %t : index
    }
    aiex.dma_free_task(%last)
  }
}
