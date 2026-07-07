//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.

// RUN: aie-opt --aie-dma-tasks-to-npu --verify-diagnostics --split-input-file %s

// A buffer descriptor that carries a rotating bd_id_window (width > 1) selects a
// different physical BD each loop iteration. Emitting that runtime-selected
// push_queue bd_id is a later increment; until then this lowering rejects a
// windowed BD with a clear diagnostic rather than miscompiling it (which would
// reuse the window base every iteration).

aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence(%arg0: memref<8xi16>) {
    // expected-note@+1 {{Error encountered while lowering this BD configuration}}
    %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      // expected-error@+1 {{rotating bd_id_window}}
      aie.dma_bd(%arg0 : memref<8xi16> offset = 0 len = 8 sizes = [] strides = []) {bd_id = 0 : i32, bd_id_window = array<i32: 0, 1>}
      aie.end
    }
    aiex.dma_start_task(%t)
  }
}
