// RUN: phy-opt %s | FileCheck %s
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @function(%arg0: memref<1024xi32>)
func.func @function(%mem: memref<1024xi32>) {
  %idx = arith.constant 0 : index

  // CHECK: memref.load
	%0 = memref.load %mem[%idx] : memref<1024xi32>
  // CHECK: memref.store
  memref.store %0, %mem[%idx] : memref<1024xi32>

  // CHECK: physical.start_load
  %h0 = physical.start_load %mem[%idx] : memref<1024xi32>
  // CHECK: physical.start_store
  %h1 = physical.start_store %0, %mem[%idx] : memref<1024xi32>
  // CHECK: physical.wait
	%1 = physical.wait(%h0) : i32
  // CHECK: physical.wait
	physical.wait(%h1) : none
  func.return
}

// CHECK: physical.buffer
%buf = physical.buffer() : memref<1024xi32>
// CHECK: physical.core @function
%pe1 = physical.core @function(%buf) : (memref<1024xi32>) -> !physical.core
// CHECK: physical.core @function
%pe2 = physical.core @function(%buf) : (memref<1024xi32>) -> !physical.core