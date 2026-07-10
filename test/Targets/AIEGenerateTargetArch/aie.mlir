//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-target-arch %s | FileCheck --match-full-lines %s
// CHECK: AIE

module {
  %01 = aie.tile(0, 1)
}
