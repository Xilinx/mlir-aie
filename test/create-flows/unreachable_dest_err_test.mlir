//===- unreachable_dest_err_test.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// A flow whose destination cannot be reached by the router exercises the
// trace-back unreachable-destination guard (no predecessor for the dest), which
// must fail the routing cleanly rather than indexing with a -1 predecessor.

// RUN: not aie-opt --aie-create-pathfinder-flows %s 2>&1 | FileCheck %s
// CHECK: error: Unable to find a legal routing

module {
  aie.device(npu1) {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    aie.flow(%t00, DMA : 0, %t01, Core : 0)
  }
}
