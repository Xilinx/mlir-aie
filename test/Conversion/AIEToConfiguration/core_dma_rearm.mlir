//===- core_dma_rearm.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// aie.dma_rearm becomes the register writes aie-rt makes to arm a channel,
// as stores the core makes to its own tile over the processor bus. The tile's
// registers sit at 0x80000 in the core's own address space, so the address of
// each store is that base plus the register's offset within the tile.

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-lower-core-dma-rearm %s | FileCheck %s

// CHECK-LABEL: @rearm
// The stores go through a declared callee rather than llvm.call_intrinsic:
// the intrinsic is the AIE backend's, and the cores reach the others the
// same way.
// CHECK:       func.func private @llvm.aie2p.write.tm(i32, !llvm.ptr)
// The core says it uses the bus, and the configuration enables it there.
// CHECK:       aie.core({{.*}}) {
// CHECK:         llvm.inttoptr
// CHECK:         func.call @llvm.aie2p.write.tm
// CHECK:       } {aie.processor_bus}
// CHECK-NOT:   aie.dma_rearm
module @rearm {
  aie.device(npu2_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %buf = aie.buffer(%tile_0_2) {sym_name = "buf"} : memref<8xi32>
    %prod = aie.lock(%tile_0_2, 0) {init = 1 : i32}
    %cons = aie.lock(%tile_0_2, 1) {init = 0 : i32}
    %core = aie.core(%tile_0_2) {
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%cons, AcquireGreaterEqual, %c1_i32)
      aie.dma_rearm(S2MM : 0) bd_id 0 {repeat_count = 0 : i32}
      aie.end
    }
    aie.mem(%tile_0_2) {
      %0 = aie.dma(S2MM, 0) {loop = false, repeat_count = 0 : i32} [{
        %c1 = arith.constant 1 : i32
        aie.use_lock(%prod, AcquireGreaterEqual, %c1)
        aie.dma_bd(%buf : memref<8xi32>) {bd_id = 0 : i32}
        aie.use_lock(%cons, Release, %c1)
      }]
      aie.end
    }
  }
}

// -----

// A channel the tile does not have cannot be armed.
module @no_such_channel {
  aie.device(npu2_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %core = aie.core(%tile_0_2) {
      // expected-error@+1 {{arms S2MM channel 9, but tile (0, 2) has}}
      aie.dma_rearm(S2MM : 9) bd_id 0
      aie.end
    }
  }
}

// -----

// The queue's repeat count is what one arming is worth, and it is 8 bits.
module @repeat_too_large {
  aie.device(npu2_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %core = aie.core(%tile_0_2) {
      // expected-error@+1 {{repeat_count must be in [0, 255]}}
      aie.dma_rearm(S2MM : 0) bd_id 0 {repeat_count = 256 : i32}
      aie.end
    }
  }
}
