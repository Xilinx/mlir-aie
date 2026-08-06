//===- objectfifo-pad-value.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s | FileCheck %s

// aie.objectfifo carries the constant pad value alongside padDimensions
// (geometry). Check the pad_value attribute round-trips on the op.

// CHECK: aie.objectfifo @of{{.*}}pad_value = 7 : i32
aie.device(npu1_1col) {
  %shim = aie.tile(0, 0)
  %mem = aie.tile(0, 1)
  aie.objectfifo @of(%mem dimensionsToStream [<size = 8, stride = 8>, <size = 8, stride = 1>], {%shim}, 2 : i32) {padDimensions = #aie<bd_pad_layout_array[<const_pad_before = 0, const_pad_after = 1>, <const_pad_before = 0, const_pad_after = 0>]>, pad_value = 7 : i32} : !aie.objectfifo<memref<64xi8>>
}
