//===- basic_alloc_memtile_error.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --split-input-file --aie-assign-buffer-addresses="alloc-scheme=basic-sequential" %s 2>&1 | FileCheck %s
// CHECK: error: 'aie.tile' op allocated buffers exceeded available memory
// CHECK: error: 'aie.tile' op Basic sequential allocation failed.
// CHECK: error: 'aie.buffer' op pre-allocated address must be aligned to tile load/store bus width when aligned attribute is set

module @test {
  aie.device(xcve2302) {
    %0 = aie.tile(3, 1)
    %b1 = aie.buffer(%0) { sym_name = "a" } : memref<132000xi32>
    aie.memtile_dma(%0) {
      aie.end
    }
  }
}

// -----

module @test_failalign{
  aie.device(npu2){
    %0 = aie.tile(3,1)
    %b1 = aie.buffer(%0){ sym_name = "a", address = 2 : i32 } : memref<1024xi8> // buffer align=true by default
    aie.memtile_dma(%0){
      aie.end
    }
  }


}