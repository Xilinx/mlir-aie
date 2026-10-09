//===- exhaustive_col0_guard_byte.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// An initialized buffer the exhaustive fallback places must still dodge the
// first word of an npu2 column-0 memtile: see the allocateTile comment on
// initGuardBytes, which rankedPlacements enforces but the exhaustive search
// did not until it clamped ranges before building its equivalence classes.
//
// Scaled from exhaustive_bank_pin_between_buffers.mlir (a memtile's banks are
// 65536 bytes, not 16384, and there are 8 of them, not 4, so "big" is grown
// to consume the rest of the tile and keep the same forcing argument):
//
//   npu2 memtile, 524288 bytes, eight banks of 65536
//   "low"        23040 .. 38656   (address-pinned)
//   "mid"        42240 .. 43008   (address-pinned)
//   free:        0 .. 23040 (23040), 38656 .. 42240 (3584), 43008 .. 524288
//
// Only the top run holds "small" (28160) or "big" (448065). "banked" must lie
// in bank 1 (65536 .. 131072):
//   below both, they need 65792 + 476225 = 541697: past the tile;
//   above both, or above "big", it starts past 519233: past bank 1.
// So "small", then "banked" flush against it, then "big": 43008, 71168, 71424,
// which is also the only arrangement exhaustive search needs to find.
//
// "init" (256 bytes, initialized) is the only buffer that fits the first free
// run (0 .. 23040), and nothing else in this design ever contests address 0,
// so the exhaustive search's placement of it is what this test checks.

// RUN: aie-opt --aie-assign-buffer-addresses %s | FileCheck %s

// CHECK-DAG: aie.buffer({{.*}}) {address = 4 : i32, {{.*}}sym_name = "init"}
// CHECK-DAG: aie.buffer({{.*}}) {address = 43008 : i32, {{.*}}sym_name = "small"}
// CHECK-DAG: aie.buffer({{.*}}) {address = 71168 : i32, mem_bank = 1 : i32, sym_name = "banked"}
// CHECK-DAG: aie.buffer({{.*}}) {address = 71424 : i32, {{.*}}sym_name = "big"}

module @exhaustive_col0_guard_byte {
  aie.device(npu2) {
    %t = aie.tile(0, 1)
    %low = aie.buffer(%t) { sym_name = "low", address = 23040 : i32 } : memref<15616xi8>
    %mid = aie.buffer(%t) { sym_name = "mid", address = 42240 : i32 } : memref<768xi8>
    %small = aie.buffer(%t) { sym_name = "small" } : memref<28160xi8>
    %banked = aie.buffer(%t) { sym_name = "banked", mem_bank = 1 : i32 } : memref<256xi8>
    %big = aie.buffer(%t) { sym_name = "big" } : memref<448065xi8>
    %init = aie.buffer(%t) { sym_name = "init" } : memref<256xi8> = dense<1>
    aie.memtile_dma(%t) {
      aie.end
    }
  }
}
