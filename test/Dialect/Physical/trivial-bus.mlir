// RUN: phy-opt %s | FileCheck %s
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @function
func.func @function(%bus: !physical.bus<i32>) {
  %address = arith.constant 10 : index

  // CHECK: physical.start_load
  %h0 = physical.start_load %bus[%address] : !physical.bus<i32>
  %v0 = physical.wait(%h0) : i32

  // CHECK: physical.start_store
  %h1 = physical.start_store %v0, %bus[%address] : !physical.bus<i32>
  physical.wait(%h1) : none

  func.return
}

// CHECK: physical.buffer
%buf = physical.buffer() : memref<1024xi32>
// CHECK: physical.bus
%bus = physical.bus() : !physical.bus<i32>
// CHECK: physical.bus_mmap
physical.bus_mmap(%bus[10:15], %buf[20:]: memref<1024xi32>)

// CHECK: physical.core @function
%pe = physical.core @function(%bus) : (!physical.bus<i32>) -> !physical.core
