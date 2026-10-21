//===- cpp_full_elf_empty_sequence.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A full ELF carries an empty runtime sequence, @aiecc_empty, after a
// device's own: the host dispatches it to time the fixed cost of a dispatch.
// --sequence-name keeps it beside the sequence it selects, and a device that
// already holds the name is refused.

// REQUIRES: peano

// RUN: rm -rf %t && mkdir -p %t/all %t/one %t/taken
// RUN: cd %t/all && aiecc --get-full-elf --tmpdir=%t/all %s 2>&1
// RUN: FileCheck %s < %t/all/full_elf_config.json
// RUN: cd %t/one && aiecc --get-full-elf --sequence-name=other --tmpdir=%t/one %s 2>&1
// RUN: FileCheck %s --check-prefix=ONE < %t/one/full_elf_config.json
// RUN: sed 's/@other/@aiecc_empty/' %s > %t/taken/taken.mlir
// RUN: cd %t/taken && not aiecc --get-full-elf --tmpdir=%t/taken %t/taken/taken.mlir 2>&1 | FileCheck %s --check-prefix=TAKEN

// CHECK: "id": "sequence"
// CHECK: "id": "other"
// CHECK: "id": "aiecc_empty"
// CHECK: "name": "main"

// ONE-NOT: "id": "sequence"
// ONE: "id": "other"
// ONE-NOT: "id": "sequence"
// ONE: "id": "aiecc_empty"
// ONE-NOT: "id": "sequence"

// TAKEN: error: 'aie.device' op holds a symbol named 'aiecc_empty', the empty runtime sequence aiecc adds to a full ELF

module {
  aie.device(npu2) @main {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_in(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %c1_i32 = arith.constant 1 : i32
      %elem_in = aie.objectfifo.acquire @of_in(Consume, 1) : memref<16xi32>
      %elem_out = aie.objectfifo.acquire @of_out(Produce, 1) : memref<16xi32>
      scf.for %i = %c0 to %c16 step %c1 {
        %val = memref.load %elem_in[%i] : memref<16xi32>
        %result = arith.addi %val, %c1_i32 : i32
        memref.store %result, %elem_out[%i] : memref<16xi32>
      }
      aie.objectfifo.release @of_in(Consume, 1)
      aie.objectfifo.release @of_out(Produce, 1)
      aie.end
    }

    aie.runtime_sequence @sequence(%in : memref<16xi32>, %out : memref<16xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c16 = arith.constant 16 : i64
      aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c16][%c0,%c0,%c0,%c1]) {metadata = @of_out, id = 1 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%in[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c16][%c0,%c0,%c0,%c1]) {metadata = @of_in, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @of_out}
    }

    aie.runtime_sequence @other(%in : memref<16xi32>, %out : memref<16xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c16 = arith.constant 16 : i64
      aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c16][%c0,%c0,%c0,%c1]) {metadata = @of_out, id = 1 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%in[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c16][%c0,%c0,%c0,%c1]) {metadata = @of_in, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @of_out}
    }
  }
}
