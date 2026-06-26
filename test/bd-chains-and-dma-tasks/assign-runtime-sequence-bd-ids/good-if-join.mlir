//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.

// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids --split-input-file %s | FileCheck %s

// A task handle yielded out of an scf.if arm and freed via the if-result join.
// This is not a loop rotation: each configure gets a single static BD id (no
// bd_id_window). Here %x is live going into the if and the else arm configures
// %z alongside it, so they need distinct ids (peak = 2): %x = 0, %z = 1. The
// single join free dma_free_task(%r) resolves to both arm configures and
// recycles their ids (the arms are mutually exclusive at run time, but both are
// statically allocated).

// CHECK-LABEL: @if_join
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @if_join(%arg0: memref<8xi16>, %c: i1) {
    // CHECK: aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 0 : i32}
    // CHECK-NOT: bd_id_window
    %x = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
      aie.end
    }
    %r = scf.if %c -> (index) {
      scf.yield %x : index
    } else {
      // CHECK: aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 1 : i32}
      %z = aiex.dma_configure_task(%tile_0_0, MM2S, 1) {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      aiex.dma_start_task(%z)
      scf.yield %z : index
    }
    aiex.dma_free_task(%r)
  }
}
