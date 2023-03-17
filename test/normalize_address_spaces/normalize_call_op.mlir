//===- normalize_call_op.mlir ----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-normalize-address-spaces | FileCheck %s
// CHECK: memref.global "public" @buffer : memref<1024xi32>
// CHECK:   %[[VAL_0:.*]] = memref.get_global @buffer : memref<1024xi32>
// CHECK:   memref.assume_alignment %[[VAL_0]], 32 : memref<1024xi32>
// CHECK:   call @external_function(%[[VAL_0]]) : (memref<1024xi32>) -> ()
// CHECK: func.func private @external_function(memref<1024xi32>)
module @aie attributes {llvm.target_triple = "aie"} {
 AIE.device(xcvc1902) {
  memref.global "public" @buffer : memref<1024xi32, 2>
  func.func @coreXY() {
    %0 = memref.get_global @buffer : memref<1024xi32, 2>
    memref.assume_alignment %0, 32 : memref<1024xi32, 2>
    AIE.nextBd ^bb1
  ^bb1:  // pred: ^bb0
    AIE.nextBd ^bb2
  ^bb2:  // pred: ^bb1
    call @external_function(%0) : (memref<1024xi32, 2>) -> ()
    return
  }
  func.func private @external_function(memref<1024xi32, 2>)
 }
}