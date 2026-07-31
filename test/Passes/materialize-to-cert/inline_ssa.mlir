//===- inline_ssa.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-materialize-runtime-sequences --convert-aie-to-transaction --aie-assign-lock-ids --aie-lower-set-lock --aie-npu-to-cert %s | FileCheck %s

// Test that SSA values like locks and tiles are inlined into the calling
// calling runtime sequence.

module {
  aie.device(npu2) {
    // The following SSA values should have been inlined from the aiex.run call, since the referenced runtime sequence references them.
    // CHECK: %[[TILE:.*]] = aie.tile(0, 2)
    // CHECK: %[[LOCK:.*]] = aie.lock(%[[TILE]], 0)
    // CHECK: %[[BUFFER:.*]] = aie.buffer(%[[TILE]])

    // The main sequence is lowered to a bare cert.job (no enclosing page yet).
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      // CHECK: aiex.cert.job
      // CHECK: aiex.cert.load_pdi(1, @other_device)
      // CHECK: aiex.npu.rtp_write(@rtp_0_0, 0, %{{.*}}) : i32
      // CHECK: aiex.cert.write32(2224128, 1)
      aiex.configure @other_device {
        aiex.run @sequence(%arg0) : (memref<64xi32>)
      }
    }
  }

  // The other_device is absorbed into a cert.section, not a separate device
  // CHECK: aiex.cert.section @other_device
  aie.device(npu2) @other_device {
    // The following are the original SSA value definitions -- ensure they are still in the device.
    // CHECK-NOT: aie.tile(0, 2)
    // CHECK-NOT: aie.buffer
    // CHECK-NOT: aie.lock
    %tile_0_2 = aie.tile(0, 2)

    %rtp_0_0 = aie.buffer(%tile_0_2) {sym_name = "rtp_0_0", address = 0xDEADBEEF : i32} : memref<1xi32>
    %lock_0_2 = aie.lock(%tile_0_2)

    aie.runtime_sequence (%arg0: memref<64xi32>) {
      // These are the operations that reference other SSA values in the device, which will require hoisting those SSA values into the calling device.
      %rtpv0 = arith.constant -1168197103 : i32
      aiex.npu.rtp_write(@rtp_0_0, 0, %rtpv0) : i32
      aiex.set_lock(%lock_0_2, 1)
    }
  }
}
