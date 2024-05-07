//===- bad_cascade.mlir ----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt -split-input-file %s 2>&1 | FileCheck %s
// CHECK: error{{.*}}'aie.cascade_flow' op tiles must be adjacent

aie.device(xcve2802) {
  %t13 = aie.tile(1, 3)
  %t33 = aie.tile(3, 3)
  aie.cascade_flow(%t13, %t33)
}

// -----

// CHECK: error{{.*}}'aie.cascade_flow' op memTile row has no cascade stream interface

aie.device(xcve2802) {
  %t12 = aie.tile(1, 2)
  %t22 = aie.tile(2, 2)
  aie.cascade_flow(%t12, %t22)
}

// -----

// CHECK: error{{.*}}'aie.cascade_flow' op shimTile row has no cascade stream interface

aie.device(npu) {
  %t10 = aie.tile(1, 0)
  %t20 = aie.tile(2, 0)
  aie.cascade_flow(%t10, %t20)
}

// -----

// CHECK: error{{.*}}'aie.cascade_flow' op memTile row has no cascade stream interface

aie.device(npu) {
  %t11 = aie.tile(1, 1)
  %t21 = aie.tile(2, 1)
  aie.cascade_flow(%t11, %t21)
}

// -----

// CHECK: error{{.*}}'aie.configure_cascade' op input direction of cascade must be North or West on AIE2

aie.device(xcve2802) {
  %t13 = aie.tile(1, 3)
  aie.configure_cascade(%t13, East, South)
}

// -----

// CHECK: error{{.*}}'aie.configure_cascade' op output direction of cascade must be South or East on AIE2

aie.device(xcve2802) {
  %t13 = aie.tile(1, 3)
  aie.configure_cascade(%t13, North, West)
}

// -----

// CHECK: error{{.*}}'aie.configure_cascade' op cascade not supported in AIE1

aie.device(xcvc1902) {
  %t13 = aie.tile(1, 3)
  aie.configure_cascade(%t13, North, East)
}

// -----

// CHECK: error{{.*}}'aie.configure_cascade' op shimTile row has no cascade stream interface

aie.device(xcve2802) {
  %t10 = aie.tile(1, 0)
  aie.configure_cascade(%t10, North, West)
}

// -----

// CHECK: error{{.*}}'aie.configure_cascade' op memTile row has no cascade stream interface

aie.device(npu) {
  %t11 = aie.tile(1, 1)
  aie.configure_cascade(%t11, North, West)
}
