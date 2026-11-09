//===- test_sa_lock_spill_turns.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-place-tiles="placer=sa_placer sa-seed=1" %s | aie-opt --aie-objectfifo-split --aie-assign-packet-ids --aie-objectfifo-allocate | FileCheck %s
// RUN: aie-opt --aie-place-tiles="placer=sa_placer sa-seed=1" %s | aie-opt --aie-objectfifo-split --aie-assign-packet-ids --aie-objectfifo-allocate | FileCheck %s --check-prefix=NOLOCK

// (1, 1) is too full for @a_in's objects, which spill to a mem tile
// allocation creates at (2, 1). The core reads three fifos over two input
// channels, so the first round runs out of channels and allocation retries
// with @b and @c taking turns. In every round @a_in's locks go on (0, 1),
// a tile that existed before, never on the spill tile; sa_lock_check.test
// checks the SA agrees.
// CHECK-DAG: %[[SPILL:.*]] = aie.tile(2, 1)
// CHECK-DAG: aie.buffer(%[[SPILL]]) {sym_name = "a_in__cons__buf0"}
// CHECK-DAG: aie.lock(%mem_tile_0_1) {init = 2 : i32, sym_name = "a_in__cons__prodlock0"}
// CHECK-DAG: turns = [@b__cons__dma, @c__cons__dma]
// NOLOCK-NOT: aie.lock(%mem_tile_2_1

module @lock_spill_turns {
  aie.device(npu2) {
    %s = aie.tile(1, 0)
    %m0 = aie.tile(0, 1)
    %m = aie.tile(1, 1)
    %t = aie.logical_tile<CoreTile>(?, ?)
    %l0 = aie.lock(%m, 0)
    %l1 = aie.lock(%m, 1)
    %l2 = aie.lock(%m, 2)
    %l3 = aie.lock(%m, 3)
    %l4 = aie.lock(%m, 4)
    %l5 = aie.lock(%m, 5)
    %l6 = aie.lock(%m, 6)
    %l7 = aie.lock(%m, 7)
    %l8 = aie.lock(%m, 8)
    %l9 = aie.lock(%m, 9)
    %l10 = aie.lock(%m, 10)
    %l11 = aie.lock(%m, 11)
    %l12 = aie.lock(%m, 12)
    %l13 = aie.lock(%m, 13)
    %l14 = aie.lock(%m, 14)
    %l15 = aie.lock(%m, 15)
    %l16 = aie.lock(%m, 16)
    %l17 = aie.lock(%m, 17)
    %l18 = aie.lock(%m, 18)
    %l19 = aie.lock(%m, 19)
    %l20 = aie.lock(%m, 20)
    %l21 = aie.lock(%m, 21)
    %l22 = aie.lock(%m, 22)
    %l23 = aie.lock(%m, 23)
    %l24 = aie.lock(%m, 24)
    %l25 = aie.lock(%m, 25)
    %l26 = aie.lock(%m, 26)
    %l27 = aie.lock(%m, 27)
    %l28 = aie.lock(%m, 28)
    %l29 = aie.lock(%m, 29)
    %l30 = aie.lock(%m, 30)
    %l31 = aie.lock(%m, 31)
    %l32 = aie.lock(%m, 32)
    %l33 = aie.lock(%m, 33)
    %l34 = aie.lock(%m, 34)
    %l35 = aie.lock(%m, 35)
    %l36 = aie.lock(%m, 36)
    %l37 = aie.lock(%m, 37)
    %l38 = aie.lock(%m, 38)
    %l39 = aie.lock(%m, 39)
    %l40 = aie.lock(%m, 40)
    %l41 = aie.lock(%m, 41)
    %l42 = aie.lock(%m, 42)
    %l43 = aie.lock(%m, 43)
    %l44 = aie.lock(%m, 44)
    %l45 = aie.lock(%m, 45)
    %l46 = aie.lock(%m, 46)
    %l47 = aie.lock(%m, 47)
    %l48 = aie.lock(%m, 48)
    %l49 = aie.lock(%m, 49)
    %l50 = aie.lock(%m, 50)
    %l51 = aie.lock(%m, 51)
    %l52 = aie.lock(%m, 52)
    %l53 = aie.lock(%m, 53)
    %l54 = aie.lock(%m, 54)
    %l55 = aie.lock(%m, 55)
    %l56 = aie.lock(%m, 56)
    %l57 = aie.lock(%m, 57)
    %l58 = aie.lock(%m, 58)
    %l59 = aie.lock(%m, 59)
    %l60 = aie.lock(%m, 60)
    %l61 = aie.lock(%m, 61)
    %fill0 = aie.buffer(%m0) {sym_name = "fill0"} : memref<130000xi32>
    %big = aie.buffer(%m) {sym_name = "big"} : memref<126000xi32>
    aie.objectfifo @a_in(%s, {%m}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @a(%m, {%t}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@a_in] -> [@a] ([] [])
    aie.objectfifo @bc_in(%s, {%m}, 2 : i32) : !aie.objectfifo<memref<2048xi32>>
    aie.objectfifo @b(%m, {%t}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @c(%m, {%t}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@bc_in] -> [@b, @c] ([] [0, 1024])
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.for %i = %c0 to %c8 step %c1 {
        %a0, %a1 = aie.objectfifo.acquire @a(Consume, 2) : memref<1024xi32>, memref<1024xi32>
        %b = aie.objectfifo.acquire @b(Consume, 1) : memref<1024xi32>
        %c = aie.objectfifo.acquire @c(Consume, 1) : memref<1024xi32>
        aie.objectfifo.release @a(Consume, 2)
        aie.objectfifo.release @b(Consume, 1)
        aie.objectfifo.release @c(Consume, 1)
      }
      aie.end
    }
  }
}
