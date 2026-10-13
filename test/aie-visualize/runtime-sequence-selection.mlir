//===- runtime-sequence-selection.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: aie-visualize
// RUN: aie-visualize --emit-dot %s | FileCheck %s --check-prefix=DEFAULT
// RUN: aie-visualize --emit-dot --runtime-sequence=second %s | FileCheck %s --check-prefix=SECOND
// RUN: not aie-visualize --emit-dot --runtime-sequence=missing %s 2>&1 | FileCheck %s --check-prefix=MISSING
// RUN: not aie-visualize --emit-dot --runtime-sequence=second --skip-runtime %s 2>&1 | FileCheck %s --check-prefix=CONFLICT

// DEFAULT: label="MM2S0"
// DEFAULT-NOT: %first_input
// DEFAULT-NOT: %second_input

// SECOND: label="MM2S0"
// SECOND: runtime_arg_0 {{.*}} label="%second_input"
// SECOND: runtime_arg_0 -> p_0_0_1_0_m
// SECOND-NOT: runtime_task_
// SECOND-NOT: %first_input

// MISSING: no aie.runtime_sequence named 'missing' in selected device
// CONFLICT: --skip-runtime and --runtime-sequence are mutually exclusive

module {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    aie.flow(%shim, DMA : 0, %core, Core : 0)
      via (%shim : DMA : 0 -> North : 0,
           %core : South : 0 -> Core : 0)

    aie.runtime_sequence @first(%first_input: memref<16xi32>) {
      %task = aiex.dma_configure_task(%shim, MM2S, 0) {
        aie.dma_bd(%first_input : memref<16xi32> offset = 0 len = 16)
        aie.end
      }
    }
    aie.runtime_sequence @second(%second_input: memref<16xi32>) attributes {
      runtime_arg_names = ["second_input"]
    } {
      %task = aiex.dma_configure_task(%shim, MM2S, 0) {
        aie.dma_bd(%second_input : memref<16xi32> offset = 0 len = 16)
        aie.end
      }
    }
  }
}
