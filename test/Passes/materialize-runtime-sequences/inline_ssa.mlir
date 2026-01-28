//===- inline_symbols.mlir -------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-materialize-runtime-sequences %s | FileCheck %s

// Test that SSA values like locks and tiles are inlined into the calling
// calling runtime sequence.

module {
  aie.device(npu2) {
    // The following SSA values should have been inlined from the aiex.run call, since the referenced runtime sequence references them.
    // CHECK: aie.tile(0, 2)
    // CHECK: aie.lock
    // CHECK: aie.buffer
    
    // CHECK-LABEL: aie.runtime_sequence
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      // CHECK: aiex.npu.load_pdi {device_ref = @other_device}
      // CHECK: aiex.npu.rtp_write
      // CHECK: aiex.set_lock
      aiex.configure @other_device {
        aiex.run @sequence(%arg0) : (memref<64xi32>)
      }
    }
  }
  
  // CHECK: aie.device(npu2) @other_device
  aie.device(npu2) @other_device {
    // The following are the original SSA value definitions -- ensure they are still in the device.
    // CHECK: aie.tile(0, 2)
    // CHECK: aie.buffer
    // CHECK: aie.lock
    %tile_0_2 = aie.tile(0, 2)
    
    %rtp_0_0 = aie.buffer(%tile_0_2) {sym_name = "rtp_0_0", address = 0xDEADBEEF : i32} : memref<1xi32> 
    %lock_0_2 = aie.lock(%tile_0_2)

    aie.runtime_sequence (%arg0: memref<64xi32>) {
      // These are the operations that reference other SSA values in the device, which will require hoisting those SSA values into the calling device.
      aiex.npu.rtp_write(@rtp_0_0, 0, -1168197103)
      aiex.set_lock(%lock_0_2, 1)
    }
  }
}
