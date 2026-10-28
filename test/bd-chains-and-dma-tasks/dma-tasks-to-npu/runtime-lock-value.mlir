//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --split-input-file --aie-dma-tasks-to-npu %s | FileCheck %s

// A runtime aie.use_lock value takes the dynamic BD-word path and is OR'd into
// the lock word. AcquireGreaterEqual is encoded negated and guarded to >= 1,
// since a field of 0 would mean "acquire == 0"; Acquire and Release are
// guarded to [0, max lock value].

// CHECK-LABEL: @memtile
// CHECK: %[[FROM1:.*]] = arith.subi %arg0, %c1_i32 : i32
// CHECK: %[[ACQOK:.*]] = arith.cmpi ule, %[[FROM1]], %c62_i32 : i32
// CHECK: cf.assert %[[ACQOK]], "a runtime lock value must be in [1:63]"
// CHECK: %[[NEG:.*]] = arith.subi %{{.*}}, %arg0 : i32
// CHECK: %[[RELOK:.*]] = arith.cmpi ule, %arg0, %c63_i32 : i32
// CHECK: cf.assert %[[RELOK]], "a runtime lock value must be in [0:63]"
// CHECK: %[[REL:.*]] = arith.andi %arg0, %{{.*}} : i32
// CHECK: arith.shli %[[REL]], %c24_i32
// CHECK: %[[ACQ:.*]] = arith.andi %[[NEG]], %{{.*}} : i32
// CHECK: arith.shli %[[ACQ]], %c8_i32
// CHECK: aiex.npu.blockwrite_values
// CHECK-NOT: aiex.npu.writebd
aie.device(npu2) {
  %tile_0_1 = aie.tile(0, 1)
  %buf = aie.buffer(%tile_0_1) {address = 0 : i32} : memref<4096xi32>
  %prod = aie.lock(%tile_0_1, 0) {init = 0 : i32}
  %cons = aie.lock(%tile_0_1, 1) {init = 0 : i32}
  aie.runtime_sequence @memtile(%uses: i32) {
    %t = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
      aie.use_lock(%cons, AcquireGreaterEqual, %uses)
      aie.dma_bd(%buf : memref<4096xi32> offset = 0 len = 4096) {bd_id = 0 : i32}
      aie.use_lock(%prod, Release, %uses)
      aie.end
    }
  }
}

// -----

// CHECK-LABEL: @coretile
// CHECK: %[[ACQOK:.*]] = arith.cmpi ule, %arg0, %{{.*}} : i32
// CHECK: cf.assert %[[ACQOK]], "a runtime lock value must be in [0:63]"
// CHECK: %[[RELOK:.*]] = arith.cmpi ule, %arg1, %{{.*}} : i32
// CHECK: cf.assert %[[RELOK]], "a runtime lock value must be in [0:63]"
// CHECK-NOT: arith.subi %{{.*}}, %arg0
// CHECK: %[[REL:.*]] = arith.andi %arg1, %{{.*}} : i32
// CHECK: arith.shli %[[REL]], %c18_i32
// CHECK: %[[ACQ:.*]] = arith.andi %arg0, %{{.*}} : i32
// CHECK: arith.shli %[[ACQ]], %c5_i32
// CHECK: aiex.npu.blockwrite_values
// CHECK-NOT: aiex.npu.writebd
aie.device(npu2) {
  %tile_0_2 = aie.tile(0, 2)
  %buf = aie.buffer(%tile_0_2) {address = 4096 : i32} : memref<1024xi32>
  %prod = aie.lock(%tile_0_2, 0) {init = 1 : i32}
  %cons = aie.lock(%tile_0_2, 1) {init = 0 : i32}
  aie.runtime_sequence @coretile(%acq: i32, %rel: i32) {
    %t = aiex.dma_configure_task(%tile_0_2, S2MM, 0) {
      aie.use_lock(%prod, Acquire, %acq)
      aie.dma_bd(%buf : memref<1024xi32> offset = 0 len = 1024) {bd_id = 0 : i32}
      aie.use_lock(%cons, Release, %rel)
      aie.end
    }
  }
}

// -----

// A BD with no len transfers its whole buffer, as on the static path.
// CHECK-LABEL: @no_len
// CHECK: aiex.npu.blockwrite_values(%{{.*}} : i32) values %c128_i32,
aie.device(npu2) {
  %tile_0_1 = aie.tile(0, 1)
  %buf = aie.buffer(%tile_0_1) {address = 0 : i32} : memref<128xi32>
  %prod = aie.lock(%tile_0_1, 0) {init = 0 : i32}
  %cons = aie.lock(%tile_0_1, 1) {init = 0 : i32}
  aie.runtime_sequence @no_len(%uses: i32) {
    %t = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
      aie.use_lock(%cons, AcquireGreaterEqual, %uses)
      aie.dma_bd(%buf : memref<128xi32>) {bd_id = 0 : i32}
      aie.use_lock(%prod, Release, %uses)
      aie.end
    }
  }
}

// -----

// Constant dims that cover the whole buffer keep the whole-buffer default.
// CHECK-LABEL: @no_len_dims
// CHECK: aiex.npu.blockwrite_values(%{{.*}} : i32) values %c128_i32,
aie.device(npu2) {
  %tile_0_1 = aie.tile(0, 1)
  %buf = aie.buffer(%tile_0_1) {address = 0 : i32} : memref<128xi32>
  %prod = aie.lock(%tile_0_1, 0) {init = 0 : i32}
  %cons = aie.lock(%tile_0_1, 1) {init = 0 : i32}
  aie.runtime_sequence @no_len_dims(%uses: i32) {
    %t = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
      aie.use_lock(%cons, AcquireGreaterEqual, %uses)
      aie.dma_bd(%buf : memref<128xi32> sizes = [8, 16] strides = [16, 1]) {bd_id = 0 : i32}
      aie.use_lock(%prod, Release, %uses)
      aie.end
    }
  }
}
