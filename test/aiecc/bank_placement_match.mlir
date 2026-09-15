//===- bank_placement_match.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A bank request the linker satisfied builds clean.
//
// The companion to bank_placement_mismatch.mlir: a check that rejected every
// annotated design would pass that one while making the feature unusable. The
// table asks for a paired resource, so either of two banks satisfies it and the
// test does not depend on which one this linker picks.

// REQUIRES: chess
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: xchesscc_wrapper aie2p -c %S/bank_placement_match_kernel.cc -o %t.d/bank_placement_match_kernel.o
// RUN: cd %t.d && aiecc --xchesscc --xbridge --get-core-elfs %s 2>&1 | FileCheck %s

// CHECK-NOT: is placed for memory bank

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<64xi8>>

    func.func private @classify(memref<64xi8>) attributes {link_with = "bank_placement_match_kernel.o"}

    %core_0_2 = aie.core(%tile_0_2) {
      %e = aie.objectfifo.acquire @of_out(Produce, 1) : memref<64xi8>
      func.call @classify(%e) : (memref<64xi8>) -> ()
      aie.objectfifo.release @of_out(Produce, 1)
      aie.end
    } { stack_size = 1024 : i32 }

    aie.runtime_sequence(%out : memref<64xi8>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c64 = arith.constant 64 : i64
      aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c64][%c0,%c0,%c0,%c1]) {metadata = @of_out, id = 1 : i64} : memref<64xi8>
      aiex.npu.dma_wait {symbol = @of_out}
    }
  }
}
