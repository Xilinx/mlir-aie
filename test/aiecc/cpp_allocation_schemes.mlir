//===- cpp_allocation_schemes.mlir -----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: peano

// Test buffer allocation scheme options

// RUN: aiecc --no-xchesscc --no-xbridge --alloc-scheme=basic-sequential --verbose %s | FileCheck %s --check-prefix=BASIC
// RUN: aiecc --no-xchesscc --no-xbridge --alloc-scheme=bank-aware --verbose %s | FileCheck %s --check-prefix=BANK

// BASIC: alloc-scheme=basic-sequential
// BASIC: Compilation completed successfully

// BANK: alloc-scheme=bank-aware
// BANK: Compilation completed successfully

module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_0 = aie.tile(0, 0)

    aie.objectfifo @fifo(%tile_0_0, {%tile_0_2}, 4 : i32) : !aie.objectfifo<memref<128xi32>>

    %core = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c128 = arith.constant 128 : index

      %subview = aie.objectfifo.acquire @fifo(Consume, 1) : !aie.objectfifosubview<memref<128xi32>>
      %elem = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<128xi32>> -> memref<128xi32>

      scf.for %i = %c0 to %c128 step %c1 {
        %val = memref.load %elem[%i] : memref<128xi32>
        memref.store %val, %elem[%i] : memref<128xi32>
      }

      aie.objectfifo.release @fifo(Consume, 1)
      aie.end
    }

    aie.runtime_sequence(%buf : memref<128xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c128 = arith.constant 128 : i64
      aiex.npu.dma_memcpy_nd(%buf[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c128][%c0,%c0,%c0,%c1]) {metadata = @fifo, id = 0 : i64, issue_token = true} : memref<128xi32>
      aiex.npu.dma_wait {symbol = @fifo}
    }
  }
}
