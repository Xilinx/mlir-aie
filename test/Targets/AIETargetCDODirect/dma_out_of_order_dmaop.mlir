//===- dma_out_of_order_dmaop.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-assign-bd-ids --aie-assign-buffer-addresses %s | aie-translate --aie-generate-cdo --cdo-debug=true 2>&1 | FileCheck %s

// Each receive BD is valid but UNCHAINED: the control-word top byte (bits 25-30)
// is 0x02 = VALID_BD only, with use_next_bd (bit 26) and next_bd (bits 27-30)
// clear. Low bytes 0x043FE0 = lock cfg.
// CHECK: Address: 0x000000000021D014 {{.*}} is: 0x02043FE0
// CHECK: Address: 0x000000000021D034 {{.*}} is: 0x02043FE0

// The S2MM ch0 control register on tile (0,2) (base 0x200000 | offset 0x1DE00)
// gets the out-of-order enable bit (bit 3 = 0x8).
// CHECK: (Write64): Address:  0x000000000021DE00 Data:  0x00000008

module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    aie.buffer @b(%t) : memref<8xi32>
    aie.lock @l0(%t, 0) { init = 1 : i32 }
    aie.lock @l1(%t, 1) { init = 0 : i32 }
    aie.mem(%t) {
      %c1 = arith.constant 1 : i32
      aie.dma(S2MM, 0) { out_of_order } [
        {
          aie.use_lock(@l0, AcquireGreaterEqual, %c1)
          aie.dma_bd_packet(0, 0)
          %b = aie.get_buffer @b : memref<8xi32>
          aie.dma_bd(%b : memref<8xi32> offset = 0 len = 4) { bd_id = 0 : i32 }
          aie.use_lock(@l1, Release, %c1)
        },
        {
          aie.use_lock(@l0, AcquireGreaterEqual, %c1)
          aie.dma_bd_packet(0, 0)
          %b_1 = aie.get_buffer @b : memref<8xi32>
          aie.dma_bd(%b_1 : memref<8xi32> offset = 4 len = 4) { bd_id = 1 : i32 }
          aie.use_lock(@l1, Release, %c1)
        }
      ]
      aie.end
    }
  }
}
