//===- cert_verify_groupid.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Group/placement ids must be valid microcontroller indices for the device.
// npu2 has 8 columns and 1 uC/column => 8 uCs (valid ids [0, 8)).

// RUN: aie-opt --aie-cert-verify --split-input-file --verify-diagnostics %s

// Negative: group id 99 is out of range.
aie.device(npu2) {
  // expected-error@+1 {{cert.attach_to_group group id 99 is not a valid uC id [0, 8)}}
  aiex.cert.attach_to_group(99) {
    aiex.cert.job(0) {
      aiex.cert.write32(0x1000, 1)
    }
  }
}

// -----

// Negative: an out-of-range placement attribute (pre-lowering IR).
aie.device(npu2) {
  // expected-error@+1 {{cert.page placement 42 is not a valid uC id [0, 8)}}
  aiex.cert.page {
    aiex.cert.job(0) {
      aiex.cert.write32(0x1000, 1)
    }
  } {placement = 42 : i32}
}

// -----

// Positive: group id 2 is a valid uC on npu2.
aie.device(npu2) {
  aiex.cert.attach_to_group(2) {
    aiex.cert.job(0) {
      aiex.cert.write32(0x1000, 1)
    }
  }
}
