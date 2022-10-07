// RUN: phy-opt --convert-layout-to-physical="device=aie" %s | FileCheck %s
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

func.func private @kernel() {
  cf.br ^bb
^bb:
  cf.br ^bb
}

%node = spatial.node @kernel(): () -> !spatial.node

layout.platform<"vck190"> {
  layout.device<"aie"> {
    
    // CHECK: physical.core @kernel1() {aie.tile = "6.3"} : () -> !physical.core
    layout.place<"tile/6.3/core">(%node: !spatial.node)

  }
}
