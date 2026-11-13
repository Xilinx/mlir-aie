//===- good-empty-sequence.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-dynamic-bd-pool %s | FileCheck %s

// A runtime sequence with no ops has a body region without a block; the pass
// leaves it as it is.

// CHECK-LABEL: aie.device(npu2)
// CHECK: aie.runtime_sequence
// CHECK-NOT: aiex.dma_bd_pool_pop

aie.device(npu2) {
  aie.runtime_sequence() {
  }
}
