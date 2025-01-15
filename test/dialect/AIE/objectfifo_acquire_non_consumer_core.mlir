//===- objectfifo_acquire_non_consumer_core.mlir ----------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s 2>&1 | FileCheck %s

// CHECK: 'aie.core' op consumer port of objectFifo accessed by core running on non-consumer tile

module @objectfifo_acquire_non_consumer_core {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile23 = aie.tile(2, 3)

    aie.objectfifo @of_0 (%tile12, {%tile23}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    %core12 = aie.core(%tile12) {
      %subview = aie.objectfifo.acquire @of_0 (Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
         
      aie.end
   }
 }
}
