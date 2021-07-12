//===- herd.mlir -----------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s | FileCheck %s
// CHECK-LABEL: module {
// CHECK:       }

module {
  %2 = AIE.tile(2, 3)
  %3 = AIE.tile(2, 2)
  AIE.flow(%2, Core : 0, %3, Core : 1)
  AIE.flow(%3, Core : 0, %3, Core : 0)
  AIE.flow(%3, Core : 1, %2, Core : 1)
}
