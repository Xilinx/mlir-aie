//===- objectfifo_via_DMA_via_shared_mem_bad.mlir ---------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s 2>&1 | FileCheck %s

// CHECK: `via_shared_mem` and `via_DMA` cannot occur together

module @objectfifo_via_DMA_via_shared_mem_bad {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)

    aie.objectfifo @of_0 (%tile12, {%tile13}, 2 : i32) {via_shared_mem = 1 : i32, via_DMA = true} : !aie.objectfifo<memref<16xi32>>
 }
}
