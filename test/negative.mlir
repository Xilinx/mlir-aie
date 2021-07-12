//===- negative.mlir -------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s | FileCheck %s
// CHECK-LABEL: module {
// CHECK:       }

module {
  %0 = "AIE.tile"() {col=2 : i32, row=3 : i32} : () -> (index)
  %1 = "AIE.switchbox"(%0) ({
    "AIE.connect"() {sourceBundle=0:i32,sourceChannel=0:i32,destBundle=3:i32,destChannel=0:i32}: () -> ()
    "AIE.connect"() {sourceBundle=1:i32,sourceChannel=1:i32,destBundle=4:i32,destChannel=0:i32}: () -> ()
    "AIE.end"() : () -> ()
  }) : (index) -> (index)
}
