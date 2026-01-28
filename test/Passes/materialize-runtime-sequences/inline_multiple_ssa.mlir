//===- inline_multiple_ssa.mlir --------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
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

    // -> For other operand types like locks, we don't do equivalence testing for now. Instead, each inlined device will generate its own copy.
    //    This is easiest to avoid conflicts (e.g., if locks used between different devices had different initial values).
    //    If we ever run into limitations (e.g., run out of lock IDs due to this), we can revisit this and optimize if needed. 
    // -> Buffers referenced in both devices have conflicting names -- check that the pass renamed the second buffer with a _0 prefix to disambiguate them.
    // CHECK: %lock_0_2 = aie.lock
    // CHECK: aie.buffer
    // CHECK-SAME: sym_name = "[[BUF1_NAME:.*]]"
    // CHECK: %lock_0_2_0 = aie.lock
    // CHECK: aie.buffer
    // CHECK-SAME: sym_name = "[[BUF1_NAME]]_0"
    
    // CHECK-LABEL: aie.runtime_sequence
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      // CHECK: aiex.npu.load_pdi {device_ref = @other_device}
      // CHECK: aiex.npu.rtp_write(@[[BUF1_NAME]]
      // CHECK: aiex.set_lock(%lock_0_2, 
      aiex.configure @other_device {
        aiex.run @sequence(%arg0) : (memref<64xi32>)
      }
      // CHECK: aiex.npu.load_pdi {device_ref = @third_device}
      // CHECK: aiex.npu.rtp_write(@[[BUF1_NAME]]_0
      // CHECK: aiex.set_lock(%lock_0_2_0, 
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
    %rtp_0_0 = aie.buffer(%tile_0_2) {sym_name = "rtp_0_0", address = 0xDEADBEEF : i32} : memref<1xi32> 
    %lock_0_2 = aie.lock(%tile_0_2)

    aie.runtime_sequence (%arg0: memref<64xi32>) {
      aiex.npu.rtp_write(@rtp_0_0, 0, -1168197103)
      aiex.set_lock(%lock_0_2, 1)
    }
  }

  // CHECK-LABEL: aie.device(npu2) @third_device
  aie.device(npu2) @third_device {
    // CHECK: aie.tile(0, 2)
    // CHECK: aie.buffer
    // CHECK: aie.lock
    %tile_0_2 = aie.tile(0, 2)
    %rtp_0_0 = aie.buffer(%tile_0_2) {sym_name = "rtp_0_0", address = 0xCAFEBABE : i32} : memref<1xi32> 
    %lock_0_2 = aie.lock(%tile_0_2)

    aie.runtime_sequence (%arg0: memref<64xi32>) {
      aiex.npu.rtp_write(@rtp_0_0, 0, -1168197103)
      aiex.set_lock(%lock_0_2, 1)
    }
  }
}
