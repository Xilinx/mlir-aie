//===- badtrace_shim-ve2302.mlir -------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt -split-input-file %s 2>&1 | FileCheck %s
// CHECK: error{{.*}} 'aie.connect' op illegal stream switch connection

aie.device(xcve2302) {
  %01 = aie.tile(2, 0)
  aie.switchbox(%01) {
    aie.connect<Trace: 0, East: 1>
  }
}

// -----

// CHECK: error{{.*}} 'aie.amsel' op illegal stream switch connection

aie.device(xcve2302) {
  %02 = aie.tile(1, 0)
  aie.switchbox(%02) {
    %94 = aie.amsel<0> (0)
    %95 = aie.masterset(North : 0, %94)
    aie.packet_rules(Trace : 1) {
      aie.rule(31, 1, %94)
    }
  }
}
