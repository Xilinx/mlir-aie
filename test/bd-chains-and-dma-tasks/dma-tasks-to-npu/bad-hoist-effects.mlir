//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --verify-diagnostics --aie-dma-tasks-to-npu %s

// Only side-effect-free scalar ops (and npu.require guards) are moved out of a
// BD block; a register write stays and is rejected.

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)

    aie.runtime_sequence(%arg0: memref<32xi32>) {
      %zero = arith.constant 0 : i32
      // expected-error@+1 {{Unsupported operation within BD block.}}
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // expected-note@+1 {{No lowering to NPU instructions available for this operation.}}
        aiex.npu.write32(%zero, %zero) : i32, i32
        aie.dma_bd(%arg0 : memref<32xi32> offset = 0 len = 32) {bd_id = 0 : i32}
        aie.end
      }
    }
  }
}
