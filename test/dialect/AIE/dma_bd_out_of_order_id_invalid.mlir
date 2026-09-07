//===- dma_bd_out_of_order_id_invalid.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// Not packet-enabled.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) : memref<4xi32>
    aie.mem(%t) {
        aie.dma_start(MM2S, 0, ^bd0, ^end)
      ^bd0:
        // expected-error@+1 {{out_of_order_id requires a packet-enabled BD}}
        aie.dma_bd(%b : memref<4xi32> offset = 0 len = 4) { bd_id = 0 : i32, out_of_order_id = 3 : i32 }
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

// ID too high.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) : memref<4xi32>
    aie.mem(%t) {
        aie.dma_start(MM2S, 0, ^bd0, ^end)
      ^bd0:
        aie.dma_bd_packet(0, 0)
        // expected-error@+1 {{out_of_order_id must be in [0, 63]}}
        aie.dma_bd(%b : memref<4xi32> offset = 0 len = 4) { bd_id = 0 : i32, out_of_order_id = 64 : i32 }
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

// ID too low.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) : memref<4xi32>
    aie.mem(%t) {
        aie.dma_start(MM2S, 0, ^bd0, ^end)
      ^bd0:
        aie.dma_bd_packet(0, 0)
        // expected-error@+1 {{out_of_order_id must be in [0, 63]}}
        aie.dma_bd(%b : memref<4xi32> offset = 0 len = 4) { bd_id = 0 : i32, out_of_order_id = -1 : i32 }
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

// Unsupported target.
module {
  aie.device(xcvc1902) {
    %t = aie.tile(2, 2)
    %b = aie.buffer(%t) : memref<4xi32>
    aie.mem(%t) {
        aie.dma_start(MM2S, 0, ^bd0, ^end)
      ^bd0:
        aie.dma_bd_packet(0, 0)
        // expected-error@+1 {{out_of_order_id is not supported on this device}}
        aie.dma_bd(%b : memref<4xi32> offset = 0 len = 4) { bd_id = 0 : i32, out_of_order_id = 3 : i32 }
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}
