//===- cpp_allocation_schemes.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: peano

// Test buffer allocation scheme options

// RUN: aiecc --no-xchesscc --no-xbridge --alloc-scheme=basic-sequential --get-input-with-addresses --verbose %s 2>&1 | FileCheck %s --check-prefix=BASIC
// RUN: aiecc --no-xchesscc --no-xbridge --alloc-scheme=bank-aware --get-input-with-addresses --verbose %s 2>&1 | FileCheck %s --check-prefix=BANK

// Each allocation scheme must produce valid buffer addresses
// (input_with_addresses), which is the artifact this test verifies.
// BASIC: ({{[0-9]+}}/{{[0-9]+}}) input_with_addresses.mlir
// BASIC: wrote edge 'input_with_addresses.mlir'

// BANK: ({{[0-9]+}}/{{[0-9]+}}) input_with_addresses.mlir
// BANK: wrote edge 'input_with_addresses.mlir'

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
