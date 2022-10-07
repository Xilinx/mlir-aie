// RUN: phy-opt --convert-layout-to-physical="device=aie" %s | FileCheck %s
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

func.func private @kernel(%Q1: !spatial.queue<memref<1024xi32>>, %Q2: !spatial.queue<memref<1024xi32>>) {
  cf.br ^bb
^bb:

  %c0 = arith.constant 0 : index
  %l0 = llvm.mlir.constant(0 : i32) : i32

  // CHECK: physical.lock_acquire<1> (%arg1)
  %0 = spatial.front(%Q1): memref<1024xi32>
  // CHECK: physical.lock_acquire<0> (%arg3)
  %1 = spatial.emplace(%Q2): memref<1024xi32>

  // CHECK: %arg0[%c0] : memref<1024xi32>
  memref.store %l0, %0[%c0]: memref<1024xi32>
  // CHECK: %arg2[%c0] : memref<1024xi32>
  memref.store %l0, %1[%c0]: memref<1024xi32>

  // CHECK: physical.lock_release<0> (%arg1)
  spatial.pop(%Q1: !spatial.queue<memref<1024xi32>>)
  // CHECK: physical.lock_release<1> (%arg3)
  spatial.push(%Q2: !spatial.queue<memref<1024xi32>>)
  cf.br ^bb
}

%Q1 = spatial.queue<1>(): !spatial.queue<memref<1024xi32>>
%Q2 = spatial.queue<1>(): !spatial.queue<memref<1024xi32>>
%node = spatial.node @kernel(%Q1, %Q2): (!spatial.queue<memref<1024xi32>>, !spatial.queue<memref<1024xi32>>) -> !spatial.node

layout.platform<"vck190"> {
  layout.device<"aie"> {
    
    // CHECK: physical.core @kernel1

    layout.place<"tile/6.3/id/1/buffer,tile/6.3/id/1/lock">(%Q1: !spatial.queue<memref<1024xi32>>)
    layout.place<"tile/6.3/id/2/buffer,tile/6.3/id/2/lock">(%Q2: !spatial.queue<memref<1024xi32>>)
    layout.place<"tile/6.3/core">(%node: !spatial.node)
    layout.route<[]>(%Q1: !spatial.queue<memref<1024xi32>> -> %node: !spatial.node)
    layout.route<[]>(%Q2: !spatial.queue<memref<1024xi32>> -> %node: !spatial.node)

  }
}
