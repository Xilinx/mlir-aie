//===- cert_verify_groupid.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Group/placement ids must be valid microcontroller indices for the device.
// xcve3858 has 36 columns and 1 uC/column => 36 uCs; the 32-bit party mask limits CERT group ids to [0, 32).

// RUN: aie-opt --aie-cert-verify --split-input-file --verify-diagnostics %s

// Negative: group id 99 is out of range.
aie.device(xcve3858) {
  // expected-error@+1 {{cert.attach_to_group group id 99 is not a valid uC id [0, 32)}}
  aiex.cert.attach_to_group(99) {
    aiex.cert.job(0) {
      aiex.cert.write32(0x1000, 1)
    }
  }
}

// -----

// Negative: an out-of-range placement attribute (pre-lowering IR).
aie.device(xcve3858) {
  // expected-error@+1 {{cert.page placement 42 is not a valid uC id [0, 32)}}
  aiex.cert.page {
    aiex.cert.job(0) {
      aiex.cert.write32(0x1000, 1)
    }
  } {placement = 42 : i32}
}

// -----

// Positive: group id 2 is a valid uC on xcve3858.
aie.device(xcve3858) {
  aiex.cert.attach_to_group(2) {
    aiex.cert.job(0) {
      aiex.cert.write32(0x1000, 1)
    }
  }
}

// -----

// Positive: AIE2 has one device-level uC, represented by group 0.
aie.device(npu1) {
  aiex.cert.attach_to_group(0) {
    aiex.cert.job(0) {
      aiex.cert.write32(0x1000, 1)
    }
  }
}

// -----

// Negative: AIE2P has only its device-level uC, so group 1 is invalid.
aie.device(npu2) {
  // expected-error@+1 {{cert.attach_to_group group id 1 is not a valid uC id [0, 1)}}
  aiex.cert.attach_to_group(1) {
    aiex.cert.job(0) {
      aiex.cert.write32(0x1000, 1)
    }
  }
}

// -----

// Negative: AIE1 has no microcontroller.
aie.device(xcvc1902) {
  // expected-error@+1 {{cert.attach_to_group group id 0 is not a valid uC id [0, 0)}}
  aiex.cert.attach_to_group(0) {
    aiex.cert.job(0) {
      aiex.cert.write32(0x1000, 1)
    }
  }
}

// -----

// Positive: the highest uC id representable by a party mask.
aie.device(xcve3858) {
  aiex.cert.attach_to_group(31) {
    aiex.cert.job(0) {
      aiex.cert.nop
    }
  }
}

// -----

// Negative: a physical uC that cannot be represented by a party mask.
aie.device(xcve3858) {
  // expected-error@+1 {{cert.attach_to_group group id 32 is not a valid uC id [0, 32)}}
  aiex.cert.attach_to_group(32) {
    aiex.cert.job(0) {
      aiex.cert.nop
    }
  }
}

// -----

// Positive: a one-column NPU partition still has its device-level uC.
aie.device(npu2_1col) {
  aiex.cert.attach_to_group(0) {
    aiex.cert.job(0) {
      aiex.cert.nop
    }
  }
}
