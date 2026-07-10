//===- simple2.mlir --------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows --aie-find-flows %s -o %t.opt
// RUN: FileCheck %s --check-prefix=CHECK1 < %t.opt
// RUN: aie-translate --aie-flows-to-json %t.opt | FileCheck %s --check-prefix=CHECK2

// CHECK1: %[[T23:.*]] = aie.tile(2, 3)
// CHECK1: %[[T32:.*]] = aie.tile(3, 2)
// CHECK1: aie.flow(%[[T23]], Core : 1, %[[T32]], DMA : 0)

// CHECK2: "total_path_length": 2

module {
  aie.device(xcvc1902) {
    %0 = aie.tile(2, 3)
    %1 = aie.tile(3, 2)
    aie.flow(%0, Core : 1, %1, DMA : 0)
  }
}
