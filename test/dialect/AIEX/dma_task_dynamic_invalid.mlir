//===- dma_task_dynamic_invalid.mlir ----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Verifier negative tests for dynamic SSA operands on aie.dma_bd.

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// dyn_sizes / dyn_strides must have the same length.
module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<128xi32>, %M: i64, %S: i64) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // expected-error @+1 {{dyn_sizes and dyn_strides must have the same number of operands}}
        aie.dma_bd(%arg0 : memref<128xi32>) dyn_sizes(%M, %S : i64, i64) dyn_strides(%S : i64) {bd_id = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// `dimensions` and `dyn_sizes` are mutually exclusive.
module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<128xi32>, %M: i64) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // expected-error @+1 {{dyn_sizes/dyn_strides and the static `dimensions` attribute are mutually exclusive}}
        aie.dma_bd(%arg0 : memref<128xi32>, 0, 128, [<size = 4, stride = 32>, <size = 32, stride = 1>]) dyn_sizes(%M : i64) dyn_strides(%M : i64) {bd_id = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// `pad_dimensions` is incompatible with any dynamic operand.
module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<128xi32>, %M: i64) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // expected-error @+1 {{pad_dimensions is incompatible with dynamic dma_bd operands}}
        aie.dma_bd(%arg0 : memref<128xi32>, 0, 128, [<size = 4, stride = 32>, <size = 32, stride = 1>], [<const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 0>]) dyn_offset(%M : i64) {bd_id = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}
