//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A mem tile dispatches the objects it receives to two cores in turn over one
// MM2S channel, and merges what the cores send back over one S2MM channel.
// Core a adds 1000 and core b adds 2000; the merge keeps arrival order, so the
// host checks each returned object against the input object it came from.

module {
  aie.device(NPUDEVICE) {
    %s = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 3)

    aie.objectfifo @in(%s, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @to_a(%m, {%a}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @to_b(%m, {%b}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in] -> [@to_a, @to_b] ([] []) {mode = #aie.link_mode<time>}

    aie.objectfifo @from_a(%a, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @from_b(%b, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%m, {%s}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@from_a, @from_b] -> [@out] ([] []) {mode = #aie.link_mode<time>}

    aie.core(%a) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
      %k = arith.constant 1000 : i32
      scf.for %i = %c0 to %c4 step %c1 {
        %x = aie.objectfifo.acquire @to_a(Consume, 1) : memref<16xi32>
        %y = aie.objectfifo.acquire @from_a(Produce, 1) : memref<16xi32>
        scf.for %j = %c0 to %c16 step %c1 {
          %v = memref.load %x[%j] : memref<16xi32>
          %w = arith.addi %v, %k : i32
          memref.store %w, %y[%j] : memref<16xi32>
        }
        aie.objectfifo.release @to_a(Consume, 1)
        aie.objectfifo.release @from_a(Produce, 1)
      }
      aie.end
    }
    aie.core(%b) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
      %k = arith.constant 2000 : i32
      scf.for %i = %c0 to %c4 step %c1 {
        %x = aie.objectfifo.acquire @to_b(Consume, 1) : memref<16xi32>
        %y = aie.objectfifo.acquire @from_b(Produce, 1) : memref<16xi32>
        scf.for %j = %c0 to %c16 step %c1 {
          %v = memref.load %x[%j] : memref<16xi32>
          %w = arith.addi %v, %k : i32
          memref.store %w, %y[%j] : memref<16xi32>
        }
        aie.objectfifo.release @to_b(Consume, 1)
        aie.objectfifo.release @from_b(Produce, 1)
      }
      aie.end
    }

    aie.runtime_sequence(%in : memref<128xi32>, %out : memref<128xi32>) {
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][1, 1, 1, 128][0, 0, 0, 1]) {metadata = @out, id = 1 : i64, issue_token = true} : memref<128xi32>
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 128][0, 0, 0, 1]) {metadata = @in, id = 0 : i64} : memref<128xi32>
      aiex.npu.dma_wait {symbol = @out}
    }
  }
}
