//===- via_shared_mem_errors_test.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-objectFifo-stateful-transform -split-input-file %s 2>&1 | FileCheck %s

// CHECK:   error: 'aie.objectfifo' op no access to shared memory module specified by `via_shared_mem`

aie.device(npu1) {
   %tile12 = aie.tile(1, 2)
   %tile33 = aie.tile(3, 3)

   aie.objectfifo @of1 (%tile12, {%tile33}, 1 : i32) {via_shared_mem = 1 : i32} : !aie.objectfifo<memref<16xi32>>
}

// -----

// CHECK:   error: 'aie.objectfifo' op no access to shared memory module specified by `via_shared_mem`

aie.device(npu1) {
   %tile12 = aie.tile(1, 2)
   %tile13 = aie.tile(1, 3)

   aie.objectfifo @of1 (%tile12, {%tile13}, 1 : i32) {via_shared_mem = 1 : i32, repeat_count = 2 : i32} : !aie.objectfifo<memref<16xi32>>
}

// -----

// CHECK:   error: 'aie.objectfifo' op no access to shared memory module specified by `via_shared_mem`

aie.device(npu1) {
   %tile12 = aie.tile(1, 2)
   %tile13 = aie.tile(1, 3)

   aie.objectfifo @of1 (%tile12 dimensionsToStream [<size = 1, stride = 1>, <size = 1, stride = 1>], {%tile13}, 1 : i32) {via_shared_mem = 1 : i32} : !aie.objectfifo<memref<16xi32>>
}
