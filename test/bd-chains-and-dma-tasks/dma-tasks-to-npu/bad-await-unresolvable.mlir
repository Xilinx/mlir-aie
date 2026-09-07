//===- bad-await-unresolvable.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-dma-tasks-to-npu --verify-diagnostics %s

// A task value that cannot be walked back to any aiex.dma_configure_task --
// here, a plain runtime_sequence argument, never produced by a configure --
// must fail cleanly (an ordinary dialect-conversion legalization failure)
// rather than crash. This guards both halves of the fix: getTaskOp() no
// longer asserts on a value with no defining op (dyn_cast_or_null), and
// resolveConfigureThroughCF's region-branch-predecessor walk terminates
// gracefully (returns null) once it runs out of predecessors instead of
// finding a configure.

aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @bad_unresolvable(%arg0: memref<1024xi32>, %task: index) {
    // expected-error@+1 {{failed to legalize operation 'aiex.dma_await_task'}}
    aiex.dma_await_task(%task)
  }
}
