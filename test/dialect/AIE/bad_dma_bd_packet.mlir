//===- bad_dma_bd_packet.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --split-input-file %s 2>&1 | FileCheck %s

module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) : memref<8xi32>
    aie.mem(%t) {
        aie.dma_start(S2MM, 0, ^bd0, ^end)
      ^bd0:
        // CHECK: Packet type field can only hold 3 bits.
        aie.dma_bd_packet(8, 0)
        aie.dma_bd(%b : memref<8xi32> offset = 0 len = 8)
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) : memref<8xi32>
    aie.mem(%t) {
        aie.dma_start(S2MM, 0, ^bd0, ^end)
      ^bd0:
        // CHECK: Packet ID field can only hold 5 bits.
        aie.dma_bd_packet(0, 32)
        aie.dma_bd(%b : memref<8xi32> offset = 0 len = 8)
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) : memref<8xi32>
    aie.mem(%t) {
        aie.dma_start(S2MM, 0, ^bd0, ^end)
      ^bd0:
        // CHECK: Packet type field can only hold 3 bits.
        aie.dma_bd_packet(-1, 0)
        aie.dma_bd(%b : memref<8xi32> offset = 0 len = 8)
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) : memref<8xi32>
    aie.mem(%t) {
        aie.dma_start(S2MM, 0, ^bd0, ^end)
      ^bd0:
        // CHECK: Packet ID field can only hold 5 bits.
        aie.dma_bd_packet(0, -1)
        aie.dma_bd(%b : memref<8xi32> offset = 0 len = 8)
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}
