//===- dma_out_of_order_s2mm.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// An out-of-order S2MM places each packet into the receive BD whose id matches
// the packet header. Out-of-order mode is set at config time; receive BDs are
// packet-enabled and emitted with use_next_bd=0.

// Receive BD ids are deliberately non-sequential (0, 3, 7) to check pinning.
// RUN: aie-opt --aie-assign-bd-ids --aie-assign-buffer-addresses %s | FileCheck %s --check-prefix=CONFIG
// CONFIG: aie.dma_start(S2MM, 0, {{.*}}, repeat_count = 3) {out_of_order}
// CONFIG: aie.dma_bd({{.*}}) {bd_id = 0 : i32, next_bd_id = 3 : i32}
// CONFIG: aie.dma_bd({{.*}}) {bd_id = 3 : i32, next_bd_id = 7 : i32}
// CONFIG: aie.dma_bd({{.*}}) {bd_id = 7 : i32}

// RUN: aie-opt --aie-assign-bd-ids --aie-assign-buffer-addresses %s | aie-translate --aie-generate-cdo --cdo-debug=true 2>&1 | FileCheck %s

// Compute tile (0,2), three receive BDs:
// - Each BD's register block is at 0x1D000 + bd_id*0x20, so the pinned ids 0/3/7
// place the blocks at 0x21D000 / 0x21D060 / 0x21D0E0 (and not 0x21D000/20/40).
// - Each control word (block + 0x14) is 0x02000000: valid (bit 25) but UNCHAINED
// (use_next_bd bit 26 clear).
// - Packet-enabled (bit 30 of block + 0x04).
// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x000000000021D000
// CHECK: Address: 0x000000000021D004 {{.*}} is: 0x40000000
// CHECK: Address: 0x000000000021D014 {{.*}} is: 0x02000000
// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x000000000021D060
// CHECK: Address: 0x000000000021D064 {{.*}} is: 0x40000000
// CHECK: Address: 0x000000000021D074 {{.*}} is: 0x02000000
// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x000000000021D0E0
// CHECK: Address: 0x000000000021D0E4 {{.*}} is: 0x40000000
// CHECK: Address: 0x000000000021D0F4 {{.*}} is: 0x02000000

// Out-of-order is enabed (bit 3), and the start queue has repeat count=0x3.
// CHECK: (Write64): Address:  0x000000000021DE00 Data:  0x00000008
// CHECK: (Write64): Address:  0x000000000021DE04 Data:  0x00030000

// Memtile (0,1), different register layout:
// - Packet-enable is bit 31 of the first word (0x80000004 = enable + len 4).
// - Out-of-order enable is bit 3 of the channel control (0x1A0600).
// CHECK: Address: 0x00000000001A0000 {{.*}} is: 0x80000004
// CHECK: Address: 0x00000000001A0020 {{.*}} is: 0x80000004
// CHECK: (Write64): Address:  0x00000000001A0600 Data:  0x00000008

module {
 aie.device(npu2) {
  %t02 = aie.tile(0, 2)
  %b02 = aie.buffer(%t02) { sym_name = "b02" } : memref<24 x i32>
  aie.mem(%t02) {
      aie.dma_start(S2MM, 0, ^c0, ^end0, repeat_count = 3) { out_of_order }
    ^c0:
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%b02 : memref<24 x i32> offset = 0 len = 4) { bd_id = 0 : i32 }
      aie.next_bd ^c1
    ^c1:
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%b02 : memref<24 x i32> offset = 8 len = 4) { bd_id = 3 : i32 }
      aie.next_bd ^c2
    ^c2:
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%b02 : memref<24 x i32> offset = 16 len = 4) { bd_id = 7 : i32 }
      aie.next_bd ^end0
    ^end0:
      aie.end
  }
  %t01 = aie.tile(0, 1)
  %b01 = aie.buffer(%t01) { sym_name = "b01" } : memref<16 x i32>
  aie.memtile_dma(%t01) {
      aie.dma_start(S2MM, 0, ^m0, ^end1, repeat_count = 2) { out_of_order }
    ^m0:
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%b01 : memref<16 x i32> offset = 0 len = 4) { bd_id = 0 : i32 }
      aie.next_bd ^m1
    ^m1:
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%b01 : memref<16 x i32> offset = 4 len = 4) { bd_id = 1 : i32 }
      aie.next_bd ^end1
    ^end1:
      aie.end
  }
 }
}
