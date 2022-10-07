// RUN: (phy-opt %s 2>&1 || true) | FileCheck %s
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

func.func @function(%data : i32) {
  func.return
}
%1 = llvm.mlir.constant(1) : i32

// CHECK-LABEL: 'physical.core' op operand {{.*}} must be {{.*}}, but got 'i32'
%pe = physical.core @function(%1) : (i32) -> !physical.core