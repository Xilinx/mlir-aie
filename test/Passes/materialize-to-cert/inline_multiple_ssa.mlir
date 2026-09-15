//===- inline_multiple_ssa.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-materialize-runtime-sequences --convert-aie-to-transaction --aie-assign-lock-ids --aie-lower-set-lock --aie-npu-to-cert %s | FileCheck %s

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
    // -> Lock IDs 0 and 1 are assigned to the two locks
    // CHECK: %lock_0_2 = aie.lock(%tile_0_2, 0)
    // CHECK: aie.buffer
    // CHECK-SAME: sym_name = "[[BUF1_NAME:rtp_0_0[_0]*]]"
    // CHECK: %lock_0_2_0 = aie.lock(%tile_0_2, 1)
    // CHECK: aie.buffer
    // CHECK-SAME: sym_name = "[[BUF2_NAME:rtp_0_0[_0]*]]"

    // The main sequence is lowered to a bare cert.job (no enclosing page yet).
    // CHECK: aiex.cert.job(
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      // CHECK: aiex.cert.load_pdi({{[0-9]+}}, @other_device)
      // CHECK-NEXT: aiex.npu.rtp_write(@{{rtp_0_0[_0]*}}
      // CHECK-NEXT: aiex.cert.write32(
      aiex.configure @other_device {
        aiex.run @sequence(%arg0) : (memref<64xi32>)
      }
      // CHECK: aiex.cert.load_pdi({{[0-9]+}}, @third_device)
      // CHECK-NEXT: aiex.npu.rtp_write(@{{rtp_0_0[_0]*}}
      // CHECK-NEXT: aiex.cert.write32(
      aiex.configure @third_device {
        aiex.run @sequence(%arg0) : (memref<64xi32>)
      }
    }
    // CHECK: }
  }

  // CHECK: aiex.cert.section @other_device {
  // CHECK-NEXT: aiex.cert.page {
  // CHECK-NEXT: aiex.cert.job(
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  aie.device(npu2) @other_device {
    %tile_0_2 = aie.tile(0, 2)
    %rtp_0_0 = aie.buffer(%tile_0_2) {sym_name = "rtp_0_0", address = 0xDEADBEEF : i32} : memref<1xi32>
    %lock_0_2 = aie.lock(%tile_0_2) {initial_value = 1 : i32}

    aie.runtime_sequence (%arg0: memref<64xi32>) {
      %rtpv0 = arith.constant -1168197103 : i32
      aiex.npu.rtp_write(@rtp_0_0, 0, %rtpv0) : i32
      aiex.set_lock(%lock_0_2, 1)
    }
  }

  // CHECK: aiex.cert.section @third_device {
  // CHECK-NEXT: aiex.cert.page {
  // CHECK-NEXT: aiex.cert.job(
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NOT: aie.device
  aie.device(npu2) @third_device {
    %tile_0_2 = aie.tile(0, 2)
    %rtp_0_0 = aie.buffer(%tile_0_2) {sym_name = "rtp_0_0", address = 0xCAFEBABE : i32} : memref<1xi32>
    %lock_0_2 = aie.lock(%tile_0_2)

    aie.runtime_sequence (%arg0: memref<64xi32>) {
      %rtpv1 = arith.constant -1168197103 : i32
      aiex.npu.rtp_write(@rtp_0_0, 0, %rtpv1) : i32
      aiex.set_lock(%lock_0_2, 1)
    }
  }
}
