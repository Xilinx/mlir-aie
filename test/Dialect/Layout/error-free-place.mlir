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
%node = spatial.node @function() : () -> !spatial.node
layout.platform<"xilinx"> {

  // CHECK: 'layout.place' op expects parent op 'layout.device'
  layout.place<"slr0">(%node: !spatial.node)
}
