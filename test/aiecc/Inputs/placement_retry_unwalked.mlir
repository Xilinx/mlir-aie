//===- placement_retry_unwalked.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Input for placement_retry_walked.test: the one mem tile has two 1 KiB
// holes, too few for a_in's and b_in's objects, so allocation fails before
// channel turns walk any pool.

module @turn_walk_unwalked {
  aie.device(npu2_1col) {
    %s = aie.logical_tile<ShimNOCTile>(0, 0)
    %u = aie.logical_tile<CoreTile>(0, 4)
    %m = aie.logical_tile<MemTile>(?, ?)
    %t = aie.logical_tile<CoreTile>(0, 2)
    %h0 = aie.buffer(%m) {address = 1024 : i32, sym_name = "hold0"} : memref<256xi32>
    %r0 = aie.buffer(%m) {address = 3072 : i32, sym_name = "rest0"} : memref<130304xi32>
    aie.objectfifo @a_in(%s, {%m}, [2 : i32, 7 : i32]) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @a(%m, {%t}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@a_in] -> [@a] ([] [])
    aie.objectfifo @b_in(%s, {%m}, [2 : i32, 6 : i32]) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @b(%m, {%t}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@b_in] -> [@b] ([] [])
    aie.objectfifo @c(%u, {%t}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.core(%u) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c42 = arith.constant 42 : index
      scf.for %i = %c0 to %c42 step %c1 {
        %y = aie.objectfifo.acquire @c(Produce, 1) : memref<64xi32>
        aie.objectfifo.release @c(Produce, 1)
      }
      aie.end
    }
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c42 = arith.constant 42 : index
      scf.for %i = %c0 to %c42 step %c1 {
        %x = aie.objectfifo.acquire @c(Consume, 1) : memref<64xi32>
        %a0 = aie.objectfifo.acquire @a(Consume, 1) : memref<64xi32>
        aie.objectfifo.release @a(Consume, 1)
        %bb = aie.objectfifo.acquire @b(Consume, 1) : memref<64xi32>
        aie.objectfifo.release @b(Consume, 1)
        aie.objectfifo.release @c(Consume, 1)
      }
      aie.end
    }
  }
}
