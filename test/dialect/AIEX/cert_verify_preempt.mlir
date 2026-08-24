//===- cert_verify_preempt.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// G-preempt: every uC must expose the same multiset of cert.preempt ids.

// RUN: aie-opt --aie-cert-verify --split-input-file --verify-diagnostics %s

// Positive: both uCs expose preempt id 0 -> OK.
aie.device(xcve3858) {
  aiex.cert.attach_to_group(0) {
    aiex.cert.job(0) {
      aiex.cert.preempt(0, @save0, @restore0)
    }
  }
  aiex.cert.attach_to_group(2) {
    aiex.cert.job(1) {
      aiex.cert.preempt(0, @save0, @restore0)
    }
  }
}

// -----

// Positive: device-level (uC 0) and group 2 both expose preempt id 0 -> OK.
aie.device(xcve3858) {
  aiex.cert.job(0) {
    aiex.cert.preempt(0, @save0, @restore0)
  }
  aiex.cert.attach_to_group(2) {
    aiex.cert.job(1) {
      aiex.cert.preempt(0, @save0, @restore0)
    }
  }
}

// -----

// Negative: uC 0 exposes preempt 0 but group 2 exposes none -> error on group 2.
aie.device(xcve3858) {
  aiex.cert.job(0) {
    aiex.cert.preempt(0, @save0, @restore0)
  }
  // expected-error@+1 {{cert.preempt ids on uC 2 differ from uC 0}}
  aiex.cert.attach_to_group(2) {
    aiex.cert.job(1) {
      aiex.cert.write32(0x2000, 43)
    }
  }
}
