//===- dma_out_of_order_locks.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// An out-of-order receive BD may carry a lock for on-chip completion or
// flow-control instead of the strict acquire+release an in-order BD needs.

// The out_of_order attribute survives and the receive BDs keep their pinned ids.
// RUN: aie-opt --aie-assign-bd-ids --aie-assign-buffer-addresses %s | FileCheck %s --check-prefix=CONFIG
// CONFIG: aie.dma_start(S2MM, 0, {{.*}}, repeat_count = 2) {out_of_order}
// CONFIG: aie.dma_bd({{.*}}) {bd_id = 3 : i32, next_bd_id = 5 : i32}
// CONFIG: aie.dma_bd({{.*}}) {bd_id = 5 : i32}

// RUN: aie-opt --aie-assign-bd-ids --aie-assign-buffer-addresses %s | aie-translate --aie-generate-cdo --cdo-debug=true 2>&1 | FileCheck %s
//
// Compute tile (0,2). The first receive BD is release-only: its lock word is
// emitted as 0x02040000. A no-lock out-of-order BD would emit 0x02000000.
// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x000000000021D060
// CHECK: Address: 0x000000000021D074 {{.*}} is: 0x02040000
//
// The second receive BD adds an external acquire (ooo_prod) on top of the
// release (ooo_cons): 0x02041FE1. Dropping the acquire would read 0x02040000.
// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x000000000021D0A0
// CHECK: Address: 0x000000000021D0B4 {{.*}} is: 0x02041FE1

// Memtile (0,1), same mechanism, different lock-register layout. The release-only
// receive BD's lock word is 0x81400040 (no-lock would emit 0x80000000).
// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x00000000001A0080
// CHECK: Address: 0x00000000001A009C {{.*}} is: 0x81400040

module {
 aie.device(npu2) {
  %t02 = aie.tile(0, 2)
  %b02 = aie.buffer(%t02) { sym_name = "b02" } : memref<16 x i32>
  %ooo_cons  = aie.lock(%t02, 0) { init = 0 : i32, sym_name = "ooo_cons" }
  %ooo_prod = aie.lock(%t02, 1) { init = 2 : i32, sym_name = "ooo_prod" }
  aie.mem(%t02) {
      aie.dma_start(S2MM, 0, ^c0, ^end0, repeat_count = 2) { out_of_order }
    ^c0:  // release-only completion counter
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%b02 : memref<16 x i32> offset = 0 len = 4) { bd_id = 3 : i32 }
      %o0 = arith.constant 1 : i32
      aie.use_lock(%ooo_cons, Release, %o0)
      aie.next_bd ^c1
    ^c1:  // acquire(ooo_prod)+release(ooo_cons) backpressure
      %o1 = arith.constant 1 : i32
      aie.use_lock(%ooo_prod, AcquireGreaterEqual, %o1)
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%b02 : memref<16 x i32> offset = 8 len = 4) { bd_id = 5 : i32 }
      %o2 = arith.constant 1 : i32
      aie.use_lock(%ooo_cons, Release, %o2)
      aie.next_bd ^end0
    ^end0:
      aie.end
  }
  %t01 = aie.tile(0, 1)
  %b01 = aie.buffer(%t01) { sym_name = "b01" } : memref<16 x i32>
  %ooo_cons_mt = aie.lock(%t01, 0) { init = 0 : i32, sym_name = "ooo_cons_mt" }
  aie.memtile_dma(%t01) {
      aie.dma_start(S2MM, 0, ^m0, ^end1, repeat_count = 2) { out_of_order }
    ^m0:  // release-only completion counter on a memtile receive BD
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%b01 : memref<16 x i32> offset = 0 len = 4) { bd_id = 4 : i32 }
      %m0c = arith.constant 1 : i32
      aie.use_lock(%ooo_cons_mt, Release, %m0c)
      aie.next_bd ^m1
    ^m1:
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%b01 : memref<16 x i32> offset = 4 len = 4) { bd_id = 6 : i32 }
      %m1c = arith.constant 1 : i32
      aie.use_lock(%ooo_cons_mt, Release, %m1c)
      aie.next_bd ^end1
    ^end1:
      aie.end
  }
 }
}
