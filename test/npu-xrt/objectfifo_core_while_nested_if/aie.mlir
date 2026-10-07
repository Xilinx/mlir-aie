//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// An scf.if nested in an scf.while core body. The while condition is constant
// true, so the object fifo locks pace the core. An i1 loop-carried value flips
// on every iteration and selects the branch, which makes the core apply one
// operation to even tiles and another one to odd tiles.
//
//===----------------------------------------------------------------------===//

module {
  aie.device(NPUDEVICE) {
    %shim = aie.tile(0, 0)
    %core_tile = aie.tile(0, 2)

    aie.objectfifo @of_in(%shim, {%core_tile}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of_out(%core_tile, {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.core(%core_tile) {
      %true = arith.constant true
      %false = arith.constant false
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %c2_i32 = arith.constant 2 : i32
      %c41_i32 = arith.constant 41 : i32

      %parity = scf.while (%odd = %false) : (i1) -> i1 {
        scf.condition(%true) %odd : i1
      } do {
      ^bb0(%odd : i1):
        %in = aie.objectfifo.acquire @of_in(Consume, 1) : memref<16xi32>
        %out = aie.objectfifo.acquire @of_out(Produce, 1) : memref<16xi32>
        scf.if %odd {
          scf.for %i = %c0 to %c16 step %c1 {
            %v = memref.load %in[%i] : memref<16xi32>
            %w = arith.muli %v, %c2_i32 : i32
            memref.store %w, %out[%i] : memref<16xi32>
          }
        } else {
          scf.for %i = %c0 to %c16 step %c1 {
            %v = memref.load %in[%i] : memref<16xi32>
            %w = arith.addi %v, %c41_i32 : i32
            memref.store %w, %out[%i] : memref<16xi32>
          }
        }
        aie.objectfifo.release @of_in(Consume, 1)
        aie.objectfifo.release @of_out(Produce, 1)
        %next_odd = arith.xori %odd, %true : i1
        scf.yield %next_odd : i1
      }

      aie.end
    }

    aie.runtime_sequence(%in : memref<128xi32>, %out : memref<128xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c128 = arith.constant 128 : i64
      aiex.npu.dma_memcpy_nd (%out[%c0, %c0, %c0, %c0][%c1, %c1, %c1, %c128][%c0, %c0, %c0, %c1]) { metadata = @of_out, id = 1 : i64 } : memref<128xi32>
      aiex.npu.dma_memcpy_nd (%in[%c0, %c0, %c0, %c0][%c1, %c1, %c1, %c128][%c0, %c0, %c0, %c1]) { metadata = @of_in, id = 0 : i64, issue_token = true } : memref<128xi32>
      aiex.npu.dma_wait { symbol = @of_out }
    }
  }
}
