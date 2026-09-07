//===- dma_bd_out_of_order_id.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A sending (MM2S) BD stamps its out-of-order id into the outgoing packet
// header; the receiving out-of-order S2MM channel places the data in that slot.

// RUN: aie-opt --aie-assign-bd-ids --aie-assign-buffer-addresses --split-input-file %s | aie-translate --aie-generate-cdo --cdo-debug=true --split-input-file 2>&1 | FileCheck %s

// out_of_order_id = 5 on a compute-tile (0,2) MM2S BD lands in the id field
// of the BD's second config word (0x21D004): the packet-enabled base 0x40000000
// gains the id (5) -> 0x45000000.
// CHECK: Address: 0x000000000021D004 {{.*}} is: 0x45000000
module {
 aie.device(npu2) {
  %t02 = aie.tile(0, 2)
  %b02 = aie.buffer(%t02) : memref<4 x i32>
  aie.mem(%t02) {
      aie.dma_start(MM2S, 0, ^c0, ^end0)
    ^c0:
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%b02 : memref<4 x i32> offset = 0 len = 4) { bd_id = 0 : i32, out_of_order_id = 5 : i32 }
      aie.next_bd ^end0
    ^end0:
      aie.end
  }
 }
}

// -----

// the same BD without out_of_order_id leaves that word at 0x40000000.
// CHECK: Address: 0x000000000021D004 {{.*}} is: 0x40000000
module {
 aie.device(npu2) {
  %t02 = aie.tile(0, 2)
  %b02 = aie.buffer(%t02) : memref<4 x i32>
  aie.mem(%t02) {
      aie.dma_start(MM2S, 0, ^c0, ^end0)
    ^c0:
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%b02 : memref<4 x i32> offset = 0 len = 4) { bd_id = 0 : i32 }
      aie.next_bd ^end0
    ^end0:
      aie.end
  }
 }
}
