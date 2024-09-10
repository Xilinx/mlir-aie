//===- simple_flows2.mlir --------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows --aie-find-flows %s -o %t.opt
// RUN: FileCheck %s --check-prefix=CHECK1 < %t.opt
// RUN: aie-translate --aie-flows-to-json %t.opt | FileCheck %s --check-prefix=CHECK2

// CHECK1: %[[T23:.*]] = aie.tile(2, 3)
// CHECK1: %[[T22:.*]] = aie.tile(2, 2)
// CHECK1: %[[T11:.*]] = aie.tile(1, 1)
// CHECK1: aie.flow(%[[T23]], Core : 0, %[[T22]], Core : 1)
// CHECK1: aie.flow(%[[T22]], Core : 0, %[[T11]], Core : 0)

// CHECK2: "total_path_length": 3

module {
  aie.device(xcvc1902) {
    %t23 = aie.tile(2, 3)
    %t22 = aie.tile(2, 2)
    %t11 = aie.tile(1, 1)
    aie.flow(%t23, Core : 0, %t22, Core : 1)
    aie.flow(%t22, Core : 0, %t11, Core : 0)
  }
}