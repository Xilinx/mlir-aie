//===- test_sa_placement_budget.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-place-tiles="placer=sa_placer sa-seed=1" %s -o %t.default 2>&1 | FileCheck %s --check-prefix=DEFAULT --allow-empty
// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-assign-buffer-addresses %t.default -o /dev/null
// RUN: aie-opt --aie-place-tiles="placer=sa_placer sa-seed=1 placement-budget=1" %s -o %t.one 2>&1 | FileCheck %s --check-prefix=ONE
// RUN: not aie-opt --aie-objectFifo-stateful-transform --aie-assign-buffer-addresses="placement-budget=1" %t.one -o /dev/null 2>&1 | FileCheck %s --check-prefix=ADDR
// RUN: not aie-opt --aie-place-tiles="placer=sa_placer placement-budget=0" %s 2>&1 | FileCheck %s --check-prefix=ZERO

// The SA lays core tiles out with the placement-budget addressing will use
// (aiecc forwards --placement-budget to both). At the default budget this
// design places and addresses; with a budget of one placement, addressing
// gives up on a tile and the SA says so. Before, the SA always used the
// default and was silent.
// DEFAULT-NOT: over capacity
// ONE: warning: SA placer's resource estimate is over capacity
// ADDR: error: {{.*}}could not be placed
// ZERO: error: placement-budget must be positive, got 0

module @budget {
  aie.device(npu2_1col) {
    %t0 = aie.logical_tile<CoreTile>(0, 2)
    %t1 = aie.logical_tile<CoreTile>(?, ?)
    %t2 = aie.logical_tile<CoreTile>(?, ?)
    %t3 = aie.logical_tile<CoreTile>(?, ?)
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    %b0 = aie.buffer(%t0) {sym_name = "b0"} : memref<6000xi8>
    %b1 = aie.buffer(%t0) {sym_name = "b1"} : memref<64xi8>
    %b2 = aie.buffer(%t0) {sym_name = "b2"} : memref<4097xi8>
    aie.objectfifo @it0(%t0, {%t0}, 2 : i32) : !aie.objectfifo<memref<12288xi8>>
    %b3 = aie.buffer(%t2) {sym_name = "b3"} : memref<1024xi8, 6>
    %b4 = aie.buffer(%t2) {sym_name = "b4"} : memref<2048xi8>
    %b5 = aie.buffer(%t2) {sym_name = "b5"} : memref<8192xi8>
    %b6 = aie.buffer(%t2) {sym_name = "b6"} : memref<12000xi8>
    aie.objectfifo @it2(%t2, {%t2}, 3 : i32) : !aie.objectfifo<memref<8192xi8>>
    %b7 = aie.buffer(%t3) {sym_name = "b7"} : memref<2048xi8>
    %b8 = aie.buffer(%t3) {sym_name = "b8"} : memref<8192xi8>
    aie.objectfifo @it3(%t3, {%t3}, 2 : i32) : !aie.objectfifo<memref<12288xi8>>
    aie.objectfifo @st0(%t0, {%t1}, 2 : i32) : !aie.objectfifo<memref<8192xi8>>
    aie.core(%t0) {
      %it0P = aie.objectfifo.acquire @it0(Produce, 1) : memref<12288xi8>
      aie.objectfifo.release @it0(Produce, 1)
      %it0C = aie.objectfifo.acquire @it0(Consume, 1) : memref<12288xi8>
      aie.objectfifo.release @it0(Consume, 1)
      %st0P = aie.objectfifo.acquire @st0(Produce, 1) : memref<8192xi8>
      aie.objectfifo.release @st0(Produce, 1)
      aie.end
    } {measured_data_ranges = array<i32: 1664, 4096>}
    aie.core(%t1) {
      %st0C = aie.objectfifo.acquire @st0(Consume, 1) : memref<8192xi8>
      aie.objectfifo.release @st0(Consume, 1)
      aie.end
    } {stack_size = 512 : i32}
    aie.core(%t2) {
      %it2P = aie.objectfifo.acquire @it2(Produce, 1) : memref<8192xi8>
      aie.objectfifo.release @it2(Produce, 1)
      %it2C = aie.objectfifo.acquire @it2(Consume, 1) : memref<8192xi8>
      aie.objectfifo.release @it2(Consume, 1)
      aie.end
    }
    aie.core(%t3) {
      %it3P = aie.objectfifo.acquire @it3(Produce, 1) : memref<12288xi8>
      aie.objectfifo.release @it3(Produce, 1)
      %it3C = aie.objectfifo.acquire @it3(Consume, 1) : memref<12288xi8>
      aie.objectfifo.release @it3(Consume, 1)
      aie.end
    }
  }
}
