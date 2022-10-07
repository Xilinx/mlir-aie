// RUN: phy-opt %s | FileCheck %s
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @func
func.func @func(%buf:  !spatial.queue<memref<1024xi32>>,
                %fifo: !spatial.queue<memref<i32>>) {
  func.return
}
// CHECK: spatial.queue
%buf    = spatial.queue<1>() : !spatial.queue<memref<1024xi32>>
// CHECK: spatial.queue
%fifo1  = spatial.queue<2>() : !spatial.queue<memref<i32>>
// CHECK: spatial.queue
%fifo2  = spatial.queue<2>() : !spatial.queue<memref<i32>>
// CHECK: spatial.bridge
%bridge = spatial.bridge(%fifo1 -> %fifo2: !spatial.queue<memref<i32>>)
// CHECK: spatial.node
%node   = spatial.node @func(%buf, %fifo1)
        : (!spatial.queue<memref<1024xi32>>, !spatial.queue<memref<i32>>)
        -> !spatial.node

// CHECK: layout.platform<"versal">
layout.platform<"versal"> {
  // CHECK: layout.device<"pl">
  layout.device<"pl"> {
    // CHECK: layout.place<"slr0">
    layout.place<"slr0">(%buf: !spatial.queue<memref<1024xi32>>)
    // CHECK: layout.route<["slr0-slr1"]>
    layout.route<["slr0-slr1"]>(
       %buf:    !spatial.queue<memref<1024xi32>>
    -> %node:   !spatial.node)
    // CHECK: layout.place<"slr1">
    layout.place<"slr1">(%node: !spatial.node)
    // CHECK: layout.route<[]>
    layout.route<[]>(
       %node:   !spatial.node
    -> %fifo1:  !spatial.queue<memref<i32>>)
    // CHECK: layout.place<"slr1">
    layout.place<"slr1">(%fifo1: !spatial.queue<memref<i32>>)
  }
  // CHECK: layout.device<"aie">
  layout.device<"aie"> {
    // CHECK: layout.place<"shimswitchbox">
    layout.place<"shimswitchbox">(%bridge: !spatial.node)
    // CHECK: layout.route<["tile70-tile71", "tile71-tile72"]>
    layout.route<["tile70-tile71", "tile71-tile72"]>(
       %bridge: !spatial.node
    -> %fifo2:  !spatial.queue<memref<i32>>)
    // CHECK: layout.place<"tile72">
    layout.place<"tile72">(%fifo2: !spatial.queue<memref<i32>>)
  }
  // CHECK: layout.route<["pl-aie"]>
  layout.route<["pl-aie"]>(
     %fifo1:  !spatial.queue<memref<i32>>
  -> %bridge: !spatial.node)
}
