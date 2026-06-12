//===- cpp_blockwrite_fusion.mlir - BD blockwrite fusion in C++ TXN -*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Characterization test for the BD blockwrite fusion in C++ TXN generation.
//
// A runtime-parameterized npu.dma_memcpy_nd inside an scf.for lowers to a BD
// blockwrite plus per-word write32 overrides (tagged with bd_group) plus an
// address_patch. The EmitC conversion must FUSE these into a single
// txn_append_blockwrite: a static BD-data array, the runtime word(s) assigned
// into the array by index, one blockwrite call, then the address_patch -- all
// emitted INSIDE the loop body. This pins that intended behavior.
//
//===----------------------------------------------------------------------===//

// REQUIRES: peano

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: aiecc --no-xchesscc --no-xbridge --aie-generate-txn-cpp \
// RUN:   --txn-cpp-name=%t.d/fusion.h --no-compile --no-link %s
// RUN: FileCheck %s --input-file=%t.d/fusion.h

// CHECK-LABEL: generate_txn_sequence
// The fused blockwrite happens inside the loop body.
// CHECK: for (size_t
// A static BD-data array literal is materialized...
// CHECK: uint32_t [[BD:v[0-9]+]][{{[0-9]+}}] = {0x
// ...the runtime word is assigned into it by index...
// CHECK: [[BD]][{{.*}}] =
// ...exactly one fused blockwrite call consumes that array...
// CHECK: aie_runtime::txn_append_blockwrite(txn, {{.*}}, [[BD]],
// CHECK: op_count++;
// ...followed by the address patch for the same BD.
// CHECK: aie_runtime::txn_append_address_patch(txn,
// CHECK: op_count++;

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    }

    aie.runtime_sequence(%out : memref<16xi32>, %n : i32) {
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c16_i32 = arith.constant 16 : i32
      %n_div_16 = arith.divui %n, %c16_i32 : i32

      %c0_idx = arith.index_cast %c0_i32 : i32 to index
      %c1_idx = arith.index_cast %c1_i32 : i32 to index
      %n_idx = arith.index_cast %n_div_16 : i32 to index

      scf.for %i = %c0_idx to %n_idx step %c1_idx {
        %c0 = arith.constant 0 : i64
        %c1 = arith.constant 1 : i64
        %c16 = arith.constant 16 : i64
        %dim = arith.extui %n_div_16 : i32 to i64
        aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0][%dim,%c1,%c1,%c16][%c0,%c0,%c0,%c1]) {metadata = @of_out, id = 1 : i64} : memref<16xi32>
      }
    }
  }
}
