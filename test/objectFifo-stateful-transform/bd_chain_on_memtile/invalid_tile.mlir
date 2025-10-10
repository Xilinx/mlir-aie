//===- bd_chain_on_memtile/invalid_tile.mlir ---*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s 2>&1 | FileCheck %s

// CHECK: `bd_chain_iter_count` is currently only supported on MemTiles

module @objectfifo_invalid_bd_chain_iter_count_no_memtile {
 aie.device(npu1) {
    %tile13 = aie.tile(1, 2)
    %tile14 = aie.tile(1, 3)

    aie.objectfifo @of_0 (%tile13, {%tile14}, 2 : i32) {bd_chain_iter_count = 5 : i32} : !aie.objectfifo<memref<16xi32>>
 }
}