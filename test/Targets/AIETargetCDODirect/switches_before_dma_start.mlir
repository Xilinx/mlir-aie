//===- switches_before_dma_start.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-cdo %s --cdo-debug=true | FileCheck %s

// The memtile MM2S lock starts at 1, so the channel streams as soon as it is
// enabled. Its route must already be configured by then.

// Memtile switch: master South:2, then slave DMA:0.
// CHECK: (Write64): Address:  0x00000000001B0024 Data:  0x80000000
// CHECK: (Write64): Address:  0x00000000001B0100 Data:  0x80000000
// Memtile MM2S0: push BD 0, then enable.
// CHECK: (Write64): Address:  0x00000000001A0634 Data:  0x00000000
// CHECK: (MaskWrite64): Address: 0x00000000001A0630  Mask: 0x00000000  Data: 0x00000001

module {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %full = aie.lock(%mem, 0) {init = 1 : i32}
    %empty = aie.lock(%mem, 1) {init = 0 : i32}
    %buf = aie.buffer(%mem) {address = 0 : i32, mem_bank = 0 : i32} : memref<4xi32> = dense<[0, 1, 2, 3]>
    %dma = aie.memtile_dma(%mem) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.use_lock(%full, AcquireGreaterEqual, %one)
      aie.dma_bd(%buf : memref<4xi32> offset = 0 len = 4) {bd_id = 0 : i32}
      aie.use_lock(%empty, Release, %one)
      aie.next_bd ^end
    ^end:
      aie.end
    }
    %sw00 = aie.switchbox(%shim) {
      aie.connect<North : 2, South : 2>
    }
    %mux00 = aie.shim_mux(%shim) {
      aie.connect<North : 2, DMA : 0>
    }
    %sw01 = aie.switchbox(%mem) {
      aie.connect<DMA : 0, South : 2>
    }
  }
}
