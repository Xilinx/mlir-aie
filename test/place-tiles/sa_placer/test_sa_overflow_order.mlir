//===- test_sa_overflow_order.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Two core tiles in column 0 overflow local memory by 1 KiB, each through
// one intratile fifo, and (0, 3), the only neighbour with room, can take one
// of the two ((0, 5) is full). The tile served first takes that room, so the
// SA serves overflowing tiles in position order: (0, 2)'s fifo moves, (0, 4)
// stays over and is charged for its KiB.

// RUN: aie-opt --aie-place-tiles='placer=sa_placer sa-seed=1' %s 2>&1 | FileCheck %s

// CHECK:     warning: SA placer's resource estimate is over capacity (penalty=30)
// CHECK:     %[[M:.*]] = aie.tile(0, 3)
// CHECK-NOT: aie.objectfifo.allocate @ib
// CHECK:     aie.objectfifo.allocate @ia(%[[M]])
// CHECK-NOT: aie.objectfifo.allocate @ib

module @two_overflows_one_neighbour {
  aie.device(npu2) {
    %a = aie.logical_tile<CoreTile>(0, 2)
    %m = aie.logical_tile<CoreTile>(0, 3)
    %b = aie.logical_tile<CoreTile>(0, 4)
    %f = aie.logical_tile<CoreTile>(0, 5)
    %wa = aie.buffer(%a) {sym_name = "wa"} : memref<14080xi32>
    %wm = aie.buffer(%m) {sym_name = "wm"} : memref<13312xi32>
    %wb = aie.buffer(%b) {sym_name = "wb"} : memref<14080xi32>
    %wf = aie.buffer(%f) {sym_name = "wf"} : memref<15360xi32>
    aie.objectfifo @ia(%a, {%a}, 1 : i32) {disable_synchronization = true} : !aie.objectfifo<memref<6144xi8>>
    aie.objectfifo @ib(%b, {%b}, 1 : i32) {disable_synchronization = true} : !aie.objectfifo<memref<6144xi8>>
    aie.core(%a) { aie.end }
    aie.core(%m) { aie.end }
    aie.core(%b) { aie.end }
    aie.core(%f) { aie.end }
  }
}
