//===- objectfifo-pad-value.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// aie.objectfifo carries the constant pad value alongside padDimensions
// (geometry). npu1 lacks the register.

// RUN: sed 's/DEVICE/npu2_1col/g' %s | aie-opt | FileCheck %s
// RUN: sed 's/DEVICE/npu1_1col/g' %s | aie-opt --verify-diagnostics

// CHECK: aie.objectfifo @of{{.*}}padValue = 7 : i32
aie.device(DEVICE) {
  %shim = aie.tile(0, 0)
  %mem = aie.tile(0, 1)
  // expected-error@+1 {{`padValue` requires the CONSTANT_PAD_VALUE register}}
  aie.objectfifo @of(%mem dimensionsToStream [<size = 8, stride = 8>, <size = 8, stride = 1>], {%shim}, 2 : i32) {padDimensions = #aie<bd_pad_layout_array[<const_pad_before = 0, const_pad_after = 1>, <const_pad_before = 0, const_pad_after = 0>]>, padValue = 7 : i32} : !aie.objectfifo<memref<64xi8>>
}
