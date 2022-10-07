// RUN: (phy-opt %s 2>&1 || true) | FileCheck %s
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

func.func @function() {
  func.return
}
%bus = physical.bus() : !physical.bus<i32>

// CHECK-LABEL: 'physical.core' op incorrect number of operands for callee
%pe = physical.core @function(%bus) : (!physical.bus<i32>) -> !physical.core