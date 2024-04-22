//===- tileop_cse.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --pass-pipeline="builtin.module(cse)" %s | FileCheck %s

// CHECK: %[[TILE1:.*]] = aie.tile(1, 1)
// CHECK-NOT: %[[TILE2:.*]] = aie.tile(1, 1)
// CHECK: %[[CORE1:.*]] = aie.core(%[[TILE1]])
// CHECK: %[[CORE2:.*]] = aie.core(%[[TILE1]])
module {
  %tile_1 = aie.tile(1, 1)
  %tile_2 = aie.tile(1, 1)
  %core_1 = aie.core(%tile_1) {
    aie.end
  }
  %core_2 = aie.core(%tile_2) {
    aie.end
  }
}
