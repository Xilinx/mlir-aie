//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.

// RUN: aie-opt --aie-test-runtime-bd-liveness --verify-diagnostics --split-input-file %s

// Liveness/interference behavior across scf.if. Arms are mutually exclusive, so
// tasks in different arms do not interfere (peak counts the larger arm, not the
// sum). A task live across an if interferes with tasks inside the taken arm.

// Task live across the if while an in-arm task is also live => peak 2.
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  // expected-remark@+1 {{bd-peak: tile(0,0) peak=2}}
  aie.runtime_sequence(%arg0: memref<8xi16>, %c: i1) {
    // expected-remark@+1 {{bd-liveness: backedges=0 kill=aiex.dma_free_task}}
    %x = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<8xi16> offset = 0 len = 8 sizes = [] strides = [])
      aie.end
    }
    aiex.dma_start_task(%x)
    scf.if %c {
      // expected-remark@+1 {{bd-liveness: backedges=0 kill=aiex.dma_free_task}}
      %y = aiex.dma_configure_task(%tile_0_0, MM2S, 1) {
        aie.dma_bd(%arg0 : memref<8xi16> offset = 0 len = 8 sizes = [] strides = [])
        aie.end
      }
      aiex.dma_start_task(%y)
      aiex.dma_free_task(%y)
    }
    aiex.dma_free_task(%x)
  }
}

// -----

// Independent tasks in each arm, each freed within its arm. Mutually exclusive
// => peak 1 (they can share a BD ID), not 2.
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  // expected-remark@+1 {{bd-peak: tile(0,0) peak=1}}
  aie.runtime_sequence(%arg0: memref<8xi16>, %c: i1) {
    scf.if %c {
      // expected-remark@+1 {{bd-liveness: backedges=0 kill=aiex.dma_free_task}}
      %y = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16> offset = 0 len = 8 sizes = [] strides = [])
        aie.end
      }
      aiex.dma_start_task(%y)
      aiex.dma_free_task(%y)
    } else {
      // expected-remark@+1 {{bd-liveness: backedges=0 kill=aiex.dma_free_task}}
      %z = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16> offset = 0 len = 8 sizes = [] strides = [])
        aie.end
      }
      aiex.dma_start_task(%z)
      aiex.dma_free_task(%z)
    }
  }
}

// -----

// Task yielded out of an if arm to the if-result, freed after. The trace
// follows the scf.if yield to the result => not leaked.
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  // expected-remark@+1 {{bd-peak: tile(0,0) peak=2}}
  aie.runtime_sequence(%arg0: memref<8xi16>, %c: i1) {
    // expected-remark@+1 {{bd-liveness: backedges=0 kill=aiex.dma_free_task}}
    %x = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<8xi16> offset = 0 len = 8 sizes = [] strides = [])
      aie.end
    }
    %r = scf.if %c -> (index) {
      scf.yield %x : index
    } else {
      // expected-remark@+1 {{bd-liveness: backedges=0 kill=aiex.dma_free_task}}
      %z = aiex.dma_configure_task(%tile_0_0, MM2S, 1) {
        aie.dma_bd(%arg0 : memref<8xi16> offset = 0 len = 8 sizes = [] strides = [])
        aie.end
      }
      aiex.dma_start_task(%z)
      scf.yield %z : index
    }
    aiex.dma_free_task(%r)
  }
}
