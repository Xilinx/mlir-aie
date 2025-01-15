//===- objectfifo_acquire_type_mismatch.mlir --------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s 2>&1 | FileCheck %s

// CHECK: 'aie.objectfifo.acquire' op ObjectFifo element and ObjectFifoSubview element must match.

module @objectfifo_acquire_type_mismatch {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile23 = aie.tile(2, 3)

    aie.objectfifo @of_0 (%tile12, {%tile23}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    %core12 = aie.core(%tile12) {
      %subview = aie.objectfifo.acquire @of_0 (Produce, 1) : !aie.objectfifosubview<memref<2x2xi32>>
         
      aie.end
   }
 }
}
