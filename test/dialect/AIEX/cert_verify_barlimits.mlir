//===- cert_verify_barlimits.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// G-barlimits: local_barrier_id in [0,15], remote_barrier_id in [1,8]; a uC may
// use at most 16 distinct local and 8 distinct remote ids.

// RUN: aie-opt --aie-cert-verify --split-input-file --verify-diagnostics %s

// Positive: local id 15 and remote id 8 are the in-range extremes -> OK.
aie.device(npu2) {
  aiex.cert.job(0) {
    aiex.cert.local_barrier(15, 2)
    aiex.cert.remote_barrier(8, 0x1)
  }
}

// -----

// Negative: local_barrier_id 16 is out of range.
aie.device(npu2) {
  aiex.cert.job(0) {
    // expected-error@+1 {{cert.local_barrier local_barrier_id 16 out of range [0, 15]}}
    aiex.cert.local_barrier(16, 2)
  }
}

// -----

// Negative: remote_barrier_id 9 is out of range.
aie.device(npu2) {
  aiex.cert.job(0) {
    // expected-error@+1 {{cert.remote_barrier remote_barrier_id 9 out of range [1, 8]}}
    aiex.cert.remote_barrier(9, 0x1)
  }
}

// -----

// Negative: a single uC uses 17 distinct local ids (0..16). The out-of-range id
// 16 also trips the range check; the budget check fires on the device.
// expected-error@+1 {{uC 0 uses 17 distinct local_barrier ids (max 16)}}
aie.device(npu2) {
  aiex.cert.job(0) {
    aiex.cert.local_barrier(0, 2)
    aiex.cert.local_barrier(1, 2)
    aiex.cert.local_barrier(2, 2)
    aiex.cert.local_barrier(3, 2)
    aiex.cert.local_barrier(4, 2)
    aiex.cert.local_barrier(5, 2)
    aiex.cert.local_barrier(6, 2)
    aiex.cert.local_barrier(7, 2)
    aiex.cert.local_barrier(8, 2)
    aiex.cert.local_barrier(9, 2)
    aiex.cert.local_barrier(10, 2)
    aiex.cert.local_barrier(11, 2)
    aiex.cert.local_barrier(12, 2)
    aiex.cert.local_barrier(13, 2)
    aiex.cert.local_barrier(14, 2)
    aiex.cert.local_barrier(15, 2)
    // expected-error@+1 {{cert.local_barrier local_barrier_id 16 out of range [0, 15]}}
    aiex.cert.local_barrier(16, 2)
  }
}
