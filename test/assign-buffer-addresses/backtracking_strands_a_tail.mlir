//===- backtracking_strands_a_tail.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The ranked first choice is the one that touches the fewest banks, and here
// that is the choice that fails. Everything is hand-sized so the arithmetic is
// checkable:
//
//   npu2 core tile, 65536 bytes, four banks of 16384
//   stack        0     .. 1024
//   "pinned"     1024  .. 44800   (address-pinned, cannot move)
//   free:        44800 .. 65536   = 20736 bytes, one run, ending mid-bank-2
//
// "mid" (12800) fits bank 3 alone at 49152, touching one bank, which beats
// starting at 44800 and touching two. But that strands 44800..49152 (4352) and
// leaves 61952..65536 (3584), and "small" needs 5000 contiguous: no room.
//
// Placing "mid" flush at 44800 costs a second bank and fits everything:
// 44800 + 12800 = 57600, then 57600 + 5000 = 62600, inside the tile.
//
// A single-pass allocator cannot reach that, because by the time "small" fails
// the choice that doomed it is already committed.

// RUN: aie-opt --aie-assign-buffer-addresses %s | FileCheck %s

// CHECK-DAG: aie.buffer({{.*}}) {address = 44800 : i32, {{.*}}sym_name = "mid"}
// CHECK-DAG: aie.buffer({{.*}}) {address = 57600 : i32, {{.*}}sym_name = "small"}

module @backtracking_strands_a_tail {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %pinned = aie.buffer(%t) { sym_name = "pinned", address = 1024 : i32 } : memref<43776xi8>
    %mid = aie.buffer(%t) { sym_name = "mid" } : memref<12800xi8>
    %small = aie.buffer(%t) { sym_name = "small" } : memref<5000xi8>
    aie.core(%t) {
      aie.end
    } { stack_size = 1024 : i32 }
  }
}
