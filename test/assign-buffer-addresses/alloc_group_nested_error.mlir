//===- alloc_group_nested_error.mlir ---------------------------*- MLIR -*-===//
//
//
//===----------------------------------------------------------------------===//

// An exempt pair must not hide a real overlap behind it. `in` sits inside `big`
// and is exempt from it (different groups, so they overlay on purpose); `past`
// starts exactly where `in` ends, so checking only the previous buffer in
// address order would clear `past` while it still overlaps `big`. The check
// compares against the furthest end seen.

// RUN: not aie-opt --aie-assign-buffer-addresses="alloc-scheme=basic-sequential" %s 2>&1 | FileCheck %s
// CHECK: error: 'aie.buffer' op buffer '"past"' at address 0x1100 overlaps with '"big"' at address 0x1000 (size: 4096 bytes)

module @nested_overlap_not_hidden {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    aie.core(%t) { aie.end }
    %big  = aie.buffer(%t) { sym_name = "big",  address = 4096 : i32, alloc_group = "a" } : memref<1024xi32>
    %in   = aie.buffer(%t) { sym_name = "in",   address = 4096 : i32, alloc_group = "b" } : memref<64xi32>
    %past = aie.buffer(%t) { sym_name = "past", address = 4352 : i32 } : memref<64xi32>
  }
}
