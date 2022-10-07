// RUN: phy-opt %s | FileCheck %s
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @function(%arg0: !spatial.queue<memref<i32>>)
func.func @function(%q: !spatial.queue<memref<i32>>) {
  // CHECK: spatial.valid
  %0 = spatial.valid(%q : !spatial.queue<memref<i32>>)
  // CHECK: spatial.full
  %1 = spatial.full(%q : !spatial.queue<memref<i32>>)
  // CHECK: spatial.front
  %accessor = spatial.front(%q) : memref<i32>
  // CHECK: spatial.emplace
  %writer = spatial.emplace(%q) : memref<i32>
  // CHECK: spatial.pop
  spatial.pop(%q: !spatial.queue<memref<i32>>)
  // CHECK: spatial.push
  spatial.push(%q: !spatial.queue<memref<i32>>)
  return
}

// CHECK: spatial.queue
%queue = spatial.queue<2>(): !spatial.queue<memref<i32>>

// CHECK: spatial.node @function
%node = spatial.node @function(%queue)
    : (!spatial.queue<memref<i32>>) -> !spatial.node