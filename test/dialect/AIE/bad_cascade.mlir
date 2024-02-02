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
