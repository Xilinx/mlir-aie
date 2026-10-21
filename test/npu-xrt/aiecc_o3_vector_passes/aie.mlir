//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Sign-extends a 16x64 i16 tile to i32, one 64-lane row per iteration. The
// row loop is the shape aiecc's -O3 vector passes rewrite: the IV-indexed
// transfers get hoisted pointers (aie-hoist-vector-transfer-pointers), the
// resulting loads become ptr post-increments (aie-vector-to-pointer-loops),
// and the v64xi16 load feeding the v64xi32 ups is split in two 512-bit
// halves (aievec-split-load-ups-chains).

module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)

    aie.objectfifo @in(%t00, {%t02}, 2 : i32) : !aie.objectfifo<memref<16x64xi16>>
    aie.objectfifo @out(%t02, {%t00}, 2 : i32) : !aie.objectfifo<memref<16x64xi32>>

    aie.core(%t02) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %pad = arith.constant 0 : i16
      %a = aie.objectfifo.acquire @in(Consume, 1) : memref<16x64xi16>
      %b = aie.objectfifo.acquire @out(Produce, 1) : memref<16x64xi32>
      scf.for %i = %c0 to %c16 step %c1 {
        %v = vector.transfer_read %a[%i, %c0], %pad {in_bounds = [true]} : memref<16x64xi16>, vector<64xi16>
        %e = arith.extsi %v : vector<64xi16> to vector<64xi32>
        vector.transfer_write %e, %b[%i, %c0] {in_bounds = [true]} : vector<64xi32>, memref<16x64xi32>
      }
      aie.objectfifo.release @in(Consume, 1)
      aie.objectfifo.release @out(Produce, 1)
      aie.end
    }

    aie.runtime_sequence(%in : memref<1024xi16>, %unused : memref<1024xi32>, %out : memref<1024xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c1024 = arith.constant 1024 : i64
      aiex.npu.dma_memcpy_nd (%out[%c0, %c0, %c0, %c0][%c1, %c1, %c1, %c1024][%c0, %c0, %c0, %c1]) { metadata = @out, id = 1 : i64 } : memref<1024xi32>
      aiex.npu.dma_memcpy_nd (%in[%c0, %c0, %c0, %c0][%c1, %c1, %c1, %c1024][%c0, %c0, %c0, %c1]) { metadata = @in, id = 0 : i64, issue_token = true } : memref<1024xi16>
      aiex.npu.dma_wait { symbol = @out }
    }
  }
}
