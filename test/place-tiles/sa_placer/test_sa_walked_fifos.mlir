//===- test_sa_walked_fifos.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-place-tiles="placer=sa_placer sa-seed=1 sa-walked-fifos=@main::@a_in" --aie-objectfifo-split --aie-objectfifo-verify --aie-assign-packet-ids --aie-objectfifo-allocate %s | FileCheck %s
// RUN: not aie-opt --aie-place-tiles="placer=sa_placer sa-seed=1" --aie-objectfifo-split --aie-objectfifo-verify --aie-assign-packet-ids --aie-objectfifo-allocate %s 2>&1 | FileCheck %s --check-prefix=DIAG

// The mem tiles of columns 0 and 1 hold two 1 KiB holes each. Allocation
// walks the pools of a_in and b_in whole (the core takes a and b in turn, and
// lcm(7, 6) descriptors are more than a channel has), and a walked pool fits
// no hole. Counted apart, their objects fit the holes beside the core, where
// the SA places them; allocation then finds no room and names the fifo.
// DIAG: error: 'aie.objectfifo.pool' op could not place buffers in accessible memory with available capacity for objectFIFO @main::@a_in
// DIAG: note: channel turns walked these objectFIFOs' mem tile pools whole: @main::@b_in, @main::@a_in

// Counted walked, the SA places the pools where whole buffers fit.
// CHECK-DAG: %a_in__cons__buf0 = aie.buffer(%mem_tile_2_1) {{.*}} : memref<448xi32>
// CHECK-DAG: %b_in__cons__buf0 = aie.buffer(%mem_tile_2_1) {{.*}} : memref<384xi32>

module @turn_walk {
  aie.device(npu2) {
    %s = aie.logical_tile<ShimNOCTile>(0, 0)
    %s1 = aie.logical_tile<ShimNOCTile>(1, 0)
    %b0 = aie.logical_tile<MemTile>(0, 1)
    %b1 = aie.logical_tile<MemTile>(1, 1)
    %m = aie.logical_tile<MemTile>(?, ?)
    %t = aie.logical_tile<CoreTile>(0, 2)
    %h0 = aie.buffer(%b0) {address = 1024 : i32, sym_name = "hold0"} : memref<256xi32>
    %r0 = aie.buffer(%b0) {address = 3072 : i32, sym_name = "rest0"} : memref<130304xi32>
    %h1 = aie.buffer(%b1) {address = 1024 : i32, sym_name = "hold1"} : memref<256xi32>
    %r1 = aie.buffer(%b1) {address = 3072 : i32, sym_name = "rest1"} : memref<130304xi32>
    aie.objectfifo @a_in(%s, {%m}, [2 : i32, 7 : i32]) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @a(%m, {%t}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@a_in] -> [@a] ([] [])
    aie.objectfifo @b_in(%s, {%m}, [2 : i32, 6 : i32]) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @b(%m, {%t}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@b_in] -> [@b] ([] [])
    aie.objectfifo @c(%s1, {%t}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
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
