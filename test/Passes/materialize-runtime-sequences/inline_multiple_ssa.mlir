//===- inline_multiple_ssa.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-materialize-runtime-sequences %s | FileCheck %s

// Test a scenario where we call multiple runtime sequences that require hoisting multiple SSA values and symbol definitions into the calling device.
// -> Symbol name collisions need to be resolved.
// -> If two SSA values across devices are equivalent (e.g. refer to the same tile), only one definition should be hoisted.

module {
  aie.device(npu2) {
    // The following SSA values should have been inlined from the aiex.run call, since the referenced runtime sequence references them.

    // -> Both @other_device and @third_device reference tile 0, 2, but we should only hoist one definition and share it across calls.
    // CHECK: %tile_0_2 = aie.tile(0, 2)
    // CHECK-NOT: aie.tile(0, 2)

    // -> For other objects like locks, we don't do equivalence testing for now. Instead, each inlined device will generate its own copy.
    //    This is easiest to avoid conflicts (e.g., if locks used between different devices had different initial values).
    //    If we ever run into limitations (e.g., run out of lock IDs due to this), we can revisit this and optimize if needed.
    // -> Conflicting names must be uniqued and each use must retain its source
    //    declaration, irrespective of symbol-cloning order.
    // CHECK-DAG: aie.buffer @[[BUF1_NAME:rtp_0_0(_[0-9]+)?]](%tile_0_2) {address = -559038737 : i32}
    // CHECK-DAG: aie.lock @[[LOCK1:lock_0_2(_[0-9]+)?]](%tile_0_2) {init = 1 : i32}
    // CHECK-DAG: aie.buffer @[[BUF2_NAME:rtp_0_0(_[0-9]+)?]](%tile_0_2) {address = -889275714 : i32}
    // CHECK-DAG: aie.lock @[[LOCK2:lock_0_2(_[0-9]+)?]](%tile_0_2){{ *$}}

    // CHECK-LABEL: aie.runtime_sequence
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      // CHECK: aiex.npu.load_pdi {device_ref = @other_device}
      // CHECK: aiex.npu.rtp_write(@[[BUF1_NAME]]
      // CHECK: aiex.set_lock(@[[LOCK1]],
      aiex.configure @other_device {
        aiex.run @sequence(%arg0) : (memref<64xi32>)
      }
      // CHECK: aiex.npu.load_pdi {device_ref = @third_device}
      // CHECK: aiex.npu.rtp_write(@[[BUF2_NAME]]
      // CHECK: aiex.set_lock(@[[LOCK2]],
      aiex.configure @third_device {
        aiex.run @sequence(%arg0) : (memref<64xi32>)
      }
    }
  }

  // CHECK-LABEL: aie.device(npu2) @other_device
  aie.device(npu2) @other_device {
    // CHECK: aie.tile(0, 2)
    // CHECK: aie.buffer
    // CHECK: aie.lock
    %tile_0_2 = aie.tile(0, 2)
    aie.buffer @rtp_0_0(%tile_0_2) {address = 0xDEADBEEF : i32} : memref<1xi32>
    aie.lock @lock_0_2(%tile_0_2) {init = 1 : i32}

    aie.runtime_sequence (%arg0: memref<64xi32>) {
      %cst_npu_0 = arith.constant -1168197103 : i32
      aiex.npu.rtp_write(@rtp_0_0, 0, %cst_npu_0) : i32
      %lock_0_2_v1 = arith.constant 1 : i32
      aiex.set_lock(@lock_0_2, %lock_0_2_v1)
    }
  }

  // CHECK-LABEL: aie.device(npu2) @third_device
  aie.device(npu2) @third_device {
    // CHECK: aie.tile(0, 2)
    // CHECK: aie.buffer
    // CHECK: aie.lock
    %tile_0_2 = aie.tile(0, 2)
    aie.buffer @rtp_0_0(%tile_0_2) {address = 0xCAFEBABE : i32} : memref<1xi32>
    aie.lock @lock_0_2(%tile_0_2)

    aie.runtime_sequence (%arg0: memref<64xi32>) {
      %cst_npu_1 = arith.constant -1168197103 : i32
      aiex.npu.rtp_write(@rtp_0_0, 0, %cst_npu_1) : i32
      %lock_0_2_v1 = arith.constant 1 : i32
      aiex.set_lock(@lock_0_2, %lock_0_2_v1)
    }
  }
}
