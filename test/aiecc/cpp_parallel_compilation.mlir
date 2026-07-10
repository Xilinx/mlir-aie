//===- cpp_parallel_compilation.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: peano

// Test parallel core compilation with -j option

// RUN: aiecc --no-xchesscc --no-xbridge --verbose -j 2 %s | FileCheck %s

// CHECK: Successfully parsed input file
// CHECK: Device 'main' with 2 core(s)
// CHECK: Compiling 2 core(s) in parallel (2 threads)
// CHECK: Compiling core
// CHECK: Compiling core
// CHECK: Generated ELF
// CHECK: Generated ELF
// CHECK: Compilation completed successfully

// Parallel per-core slicing must emit the same object as serial compilation.
// Compile the same design both ways into separate tmpdirs and diff the per-core
// objects: byte-identical objects pin the slicing equivalence claim.
// RUN: aiecc --no-xchesscc --no-xbridge --tmpdir=%t.ser %s
// RUN: aiecc --no-xchesscc --no-xbridge -j 2 --tmpdir=%t.par %s
// RUN: diff %t.ser/main_core_0_2.o %t.par/main_core_0_2.o
// RUN: diff %t.ser/main_core_1_2.o %t.par/main_core_1_2.o

module {
  aie.device(npu2_4col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)

    aie.objectfifo @in(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @mid(%tile_0_2, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%tile_1_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    // First core
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index

      %subview_in = aie.objectfifo.acquire @in(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
      %elem_in = aie.objectfifo.subview.access %subview_in[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>

      %subview_out = aie.objectfifo.acquire @mid(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
      %elem_out = aie.objectfifo.subview.access %subview_out[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>

      scf.for %i = %c0 to %c16 step %c1 {
        %val = memref.load %elem_in[%i] : memref<16xi32>
        memref.store %val, %elem_out[%i] : memref<16xi32>
      }

      aie.objectfifo.release @in(Consume, 1)
      aie.objectfifo.release @mid(Produce, 1)
      aie.end
    }

    // Second core
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index

      %subview_in = aie.objectfifo.acquire @mid(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
      %elem_in = aie.objectfifo.subview.access %subview_in[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>

      %subview_out = aie.objectfifo.acquire @out(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
      %elem_out = aie.objectfifo.subview.access %subview_out[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>

      scf.for %i = %c0 to %c16 step %c1 {
        %val = memref.load %elem_in[%i] : memref<16xi32>
        memref.store %val, %elem_out[%i] : memref<16xi32>
      }

      aie.objectfifo.release @mid(Consume, 1)
      aie.objectfifo.release @out(Produce, 1)
      aie.end
    }

    aie.runtime_sequence(%in : memref<16xi32>, %out : memref<16xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c16 = arith.constant 16 : i64
      aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c16][%c0,%c0,%c0,%c1]) {metadata = @out, id = 1 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%in[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c16][%c0,%c0,%c0,%c1]) {metadata = @in, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @out}
    }
  }
}
