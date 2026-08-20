//===- dma_out_of_order_s2mm_shim.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// An out-of-order channel selects a receive BD by a packet-header lookup id
// whose bits alias the task-complete-token format, so out-of-order and token
// issue are mutually exclusive. This test contrasts two S2MM channels on ONE
// shim tile: the in-order channel sets the token-issue bit, the out-of-order
// channel must clear it.

// RUN: aie-opt --aie-assign-bd-ids --aie-assign-buffer-addresses %s | aie-translate --aie-generate-cdo --cdo-debug=true 2>&1 | FileCheck %s

// Out-of-order channel 0:
// - Channel control (0x1D200) enabled out-of-order (bit 3)
// - Start queue clears token issue (bit 31): 0x00020000 = repeat 2.
// CHECK: (Write64): Address:  0x000000000001D200 Data:  0x00000008
// CHECK: (Write64): Address:  0x000000000001D204 Data:  0x00020000
// In-order channel 1 on the same tile DOES arm the token:
// - (0x80020002 = token | repeat 2 | start bd 2).
// CHECK: (Write64): Address:  0x000000000001D20C Data:  0x80020002

module {
 aie.device(npu2) {
  %t00 = aie.tile(0, 0)
  %buf = aie.external_buffer { sym_name = "buf" } : memref<16 x i32>
  %l0 = aie.lock(%t00, 2) { init = 1 : i32 }
  %l1 = aie.lock(%t00, 3) { init = 0 : i32 }
  aie.shim_dma(%t00) {
      aie.dma_start(S2MM, 0, ^c0, ^dma1, repeat_count = 2) { out_of_order }
    ^c0:
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%buf : memref<16 x i32> offset = 0 len = 4) { bd_id = 0 : i32 }
      aie.next_bd ^c1
    ^c1:
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%buf : memref<16 x i32> offset = 4 len = 4) { bd_id = 1 : i32 }
      aie.next_bd ^end
    ^dma1:  // in-order control channel: only out_of_order differs from ch0
      aie.dma_start(S2MM, 1, ^io, ^end, repeat_count = 2)
    ^io:
      %c1v = arith.constant 1 : i32
      aie.use_lock(%l0, AcquireGreaterEqual, %c1v)
      aie.dma_bd(%buf : memref<16 x i32> offset = 0 len = 4) { bd_id = 2 : i32 }
      aie.use_lock(%l1, Release, %c1v)
      aie.next_bd ^end
    ^end:
      aie.end
  }
 }
}
