//===- bd_chain_on_memtile/bad_iterations.mlir ----------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s 2>&1 | FileCheck %s

// CHECK: `iter_count` must be between 1 and 256
// CHECK: `iter_count` must be between 1 and 256

module {
  aie.device(npu1_1col) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    aie.objectfifo @in(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) {iter_count = 0 : i32} : !aie.objectfifo<memref<1024xi32>> 
    aie.objectfifo @in_fwd(%mem_tile_0_1, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<1024xi32>> 
    aie.objectfifo.link [@in] -> [@in_fwd]([] [0])
  }
}

module {
  aie.device(npu1_1col) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    aie.objectfifo @in(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) {iter_count = 257 : i32} : !aie.objectfifo<memref<1024xi32>> 
    aie.objectfifo @in_fwd(%mem_tile_0_1, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<1024xi32>> 
    aie.objectfifo.link [@in] -> [@in_fwd]([] [0])
  }
}

