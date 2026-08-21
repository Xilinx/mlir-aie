//===- peano_compat_gep_nuw.mlir - downgradeIRForPeano keeps GEP nuw ------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The other peano_compat tests pin a spelling Peano rejects. This one pins the
// inverse: Peano parses `getelementptr inbounds nuw`, so the downgrade must
// leave it alone. Its LLParser has taken the GEP no-wrap flags since it gained
// GEPNoWrapFlags, and the released 21.0.0 accepts the form through both opt and
// llc; stripping it dropped the flag from every objectFIFO state access for no
// reason. Should a later Peano lose the flag, this fails in its own opt.
//
// The nuw GEPs are the objectFIFO index state: two fifos put the selectors at
// distinct offsets in the state global, so the accesses carry a nonzero index.

// REQUIRES: peano

// RUN: rm -rf %t && mkdir -p %t
// RUN: aiecc --tmpdir %t %s
// RUN: FileCheck %s --input-file %t/peano-compat_main_core_0_2.ll

// CHECK: define void @core_0_2()
// CHECK: getelementptr inbounds nuw

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_in(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
      %c1_i32 = arith.constant 1 : i32

      // Looping the acquire/release keeps the fifo selectors live, so the
      // state accesses survive into the emitted IR.
      scf.for %iter = %c0 to %c4 step %c1 {
        %elem_in = aie.objectfifo.acquire @of_in (Consume, 1) : memref<16xi32>

        %elem_out = aie.objectfifo.acquire @of_out (Produce, 1) : memref<16xi32>

        scf.for %i = %c0 to %c16 step %c1 {
          %val = memref.load %elem_in[%i] : memref<16xi32>
          %result = arith.addi %val, %c1_i32 : i32
          memref.store %result, %elem_out[%i] : memref<16xi32>
        }

        aie.objectfifo.release @of_in (Consume, 1)
        aie.objectfifo.release @of_out (Produce, 1)
      }
      aie.end
    }

    aie.runtime_sequence(%in : memref<16xi32>, %out : memref<16xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c16 = arith.constant 16 : i64
      aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c16][%c0,%c0,%c0,%c1]) {metadata = @of_out, id = 1 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%in[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c16][%c0,%c0,%c0,%c1]) {metadata = @of_in, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @of_out}
    }
  }
}
