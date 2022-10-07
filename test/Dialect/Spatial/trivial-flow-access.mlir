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
  %address = arith.constant 0 : index

  %accessor = spatial.front(%q) : memref<i32>

  // CHECK: spatial.start_load
  %h0 = spatial.start_load %accessor[%address] : memref<i32>
  %v0 = spatial.wait(%h0) : i32

  // CHECK: spatial.start_store
  %h1 = spatial.start_store %v0, %accessor[%address] : memref<i32>
  spatial.wait(%h1) : none

  func.return
}

// CHECK: spatial.queue
%queue = spatial.queue<2>(): !spatial.queue<memref<i32>>

// CHECK: spatial.node @function
%node = spatial.node @function(%queue)
    : (!spatial.queue<memref<i32>>) -> !spatial.node
