//===- stack_size_before_tile_placement.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The SA placer budgets each core tile's stack. A 60000-byte buffer fits in an
// npu2 tile's 64 KiB beside the 1024-byte default stack, but not beside 8192.
// The placer sees --default-stack-size, and not a measured_stack_size the
// input carries from an earlier build.

// RUN: rm -rf %t.d && mkdir -p %t.d && cd %t.d
// RUN: %aiecc --placer=sa_placer --sa-seed=42 --get=placed.mlir --output-dir=%t.stale %s 2>&1 | FileCheck %s --check-prefix=STALE --allow-empty
// RUN: sed 's/ {measured_stack_size = 8192 : i32}//' %s > %t.d/plain.mlir
// RUN: %aiecc --placer=sa_placer --sa-seed=42 --default-stack-size=8192 --get=placed.mlir --output-dir=%t.default plain.mlir 2>&1 | FileCheck %s --check-prefix=DEFAULT

// STALE-NOT: over capacity
// DEFAULT: warning: SA placer's resource estimate is over capacity

module {
  aie.device(npu2) {
    %c = aie.logical_tile<CoreTile>(?, ?)
    %b = aie.buffer(%c) : memref<15000xi32>
    aie.core(%c) { aie.end } {measured_stack_size = 8192 : i32}
  }
}
