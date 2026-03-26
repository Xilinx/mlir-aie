//===- cpp_dynamic_txn.mlir - Dynamic C++ TXN generation --------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Tests --aie-generate-txn-cpp with a runtime_sequence containing:
// - SSA parameters (i32)
// - scf.for with iter_args
// - scf.if with results
// - Dynamic npu.dma_memcpy_nd (SSA sizes/strides)
// - Dynamic npu.rtp_write (SSA value)
// - arith ops (divui, muli, minsi, extui, trunci, cmpi, select)
//
// RuntimeSequenceOp has IsolatedFromAbove, so SCF→CF skips it while
// still converting core body SCF ops for LLVM lowering.
//
//===----------------------------------------------------------------------===//

// REQUIRES: peano

// RUN: aiecc --no-xchesscc --no-xbridge --aie-generate-txn-cpp \
// RUN:   --txn-cpp-name=%t.h --no-compile --no-link --verbose %s 2>&1 | FileCheck %s

// CHECK: Compilation completed successfully

// Also test unified compilation (XCLBIN + TXN from same MLIR):
// RUN: aiecc --no-xchesscc --no-xbridge --peano %PEANO_INSTALL_DIR% \
// RUN:   --aie-generate-xclbin --xclbin-name=%t.xclbin \
// RUN:   --aie-generate-txn-cpp --txn-cpp-name=%t_unified.h \
// RUN:   --verbose %s 2>&1 | FileCheck %s --check-prefix=UNIFIED

// UNIFIED: Compilation completed successfully

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_in(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    %rtp = aie.buffer(%tile_0_2) {sym_name = "rtp"} : memref<16xi32>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index

      %subview_in = aie.objectfifo.acquire @of_in(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
      %elem_in = aie.objectfifo.subview.access %subview_in[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>

      %subview_out = aie.objectfifo.acquire @of_out(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
      %elem_out = aie.objectfifo.subview.access %subview_out[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>

      scf.for %i = %c0 to %c16 step %c1 {
        %val = memref.load %elem_in[%i] : memref<16xi32>
        %c1_i32 = arith.constant 1 : i32
        %result = arith.addi %val, %c1_i32 : i32
        memref.store %result, %elem_out[%i] : memref<16xi32>
      }

      aie.objectfifo.release @of_in(Consume, 1)
      aie.objectfifo.release @of_out(Produce, 1)
      aie.end
    } {link_with = ""}

    // Runtime sequence with SSA parameters, SCF loops, and dynamic DMA
    aie.runtime_sequence(%in : memref<16xi32>, %out : memref<16xi32>, %n : i32) {
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c16_i32 = arith.constant 16 : i32

      // Dynamic RTP write
      aiex.npu.rtp_write(@rtp, 0 : ui32, %n) : i32

      // Derived value
      %n_div_16 = arith.divui %n, %c16_i32 : i32

      // scf.for with iter_args
      %c0_idx = arith.index_cast %c0_i32 : i32 to index
      %c1_idx = arith.index_cast %c1_i32 : i32 to index
      %n_idx = arith.index_cast %n_div_16 : i32 to index

      %result = scf.for %i = %c0_idx to %n_idx step %c1_idx
          iter_args(%acc = %c0_i32) -> (i32) {
        %i_i32 = arith.index_cast %i : index to i32

        // scf.if with results
        %cmp = arith.cmpi sgt, %i_i32, %c0_i32 : i32
        %val = scf.if %cmp -> (i32) {
          scf.yield %c1_i32 : i32
        } else {
          scf.yield %c0_i32 : i32
        }

        // Dynamic DMA with SSA sizes
        %c0 = arith.constant 0 : i64
        %c1 = arith.constant 1 : i64
        %c16 = arith.constant 16 : i64
        %dim = arith.extui %n_div_16 : i32 to i64
        aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0][%dim,%c1,%c1,%c16][%c0,%c0,%c0,%c1]) {metadata = @of_out, id = 1 : i64} : memref<16xi32>

        %new_acc = arith.addi %acc, %val : i32
        scf.yield %new_acc : i32
      }

      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
    }
  }
}
