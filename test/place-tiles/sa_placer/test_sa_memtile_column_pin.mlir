//===- test_sa_memtile_column_pin.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A logical mem tile pinned to column 3 only starts, and stays, in column 3
// under the SA, as under the sequential placer.

// RUN: aie-opt --aie-place-tiles='placer=sa_placer sa-seed=1' %s | FileCheck %s
// RUN: aie-opt --aie-place-tiles='placer=sa_placer sa-seed=2' %s | FileCheck %s
// RUN: aie-opt --aie-place-tiles %s | FileCheck %s

// CHECK: %[[M:.*]] = aie.tile(3, 1)
// CHECK: aie.flow(%[[M]], DMA : 1, %{{.*}}, DMA : 0)
module @one_pinned {
  aie.device(npu1) {
    %mem_b = aie.logical_tile<MemTile>(3, ?)
    %core_b = aie.tile(1, 2)
    aie.flow(%mem_b, DMA : 1, %core_b, DMA : 0)
  }
}
