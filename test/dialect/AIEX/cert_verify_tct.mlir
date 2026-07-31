//===- cert_verify_tct.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// G-tct: within one page, at most one *job* may wait on a given (tile_id,
// channel_id). The rule is about concurrency, so sequential waits -- two in the
// same job, or two on different pages of a uC -- are legal.

// RUN: aie-opt --aie-cert-verify --split-input-file --verify-diagnostics %s

// Positive: two wait_tcts on different (tile, channel) actors -> OK.
aie.device(npu2) {
  aiex.cert.page {
    aiex.cert.job(0) {
      aiex.cert.wait_tcts(0, 0, 1)
      aiex.cert.wait_tcts(0, 1, 1)
    }
  }
}

// -----

// Positive: two wait_tcts on the same actor in the SAME job -> OK. A job is one
// thread of control, so they run in sequence and only one is outstanding. This
// is what a runtime sequence with two npu.sync ops on one channel lowers to.
aie.device(npu2) {
  aiex.cert.page {
    aiex.cert.job(0) {
      aiex.cert.wait_tcts(0, 0, 1)
      aiex.cert.wait_tcts(0, 0, 1)
    }
  }
}

// -----

// Positive: same actor waited on from two different PAGES of one uC -> OK.
// Pages execute strictly in order, so the first wait has retired before the
// second page starts.
aie.device(npu2) {
  aiex.cert.page {
    aiex.cert.job(0) {
      aiex.cert.wait_tcts(0, 0, 1)
    }
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.wait_tcts(0, 0, 1)
    }
  }
}

// -----

// Positive: same (tile, channel) but in different uCs -> OK (per-uC scope).
aie.device(npu2) {
  aiex.cert.attach_to_group(0) {
    aiex.cert.job(0) {
      aiex.cert.wait_tcts(0, 0, 1)
    }
  }
  aiex.cert.attach_to_group(2) {
    aiex.cert.job(1) {
      aiex.cert.wait_tcts(0, 0, 1)
    }
  }
}

// -----

// Negative: two jobs cooperatively scheduled on ONE page both wait on the same
// (tile, channel) -> both waits can be outstanding at once -> error.
aie.device(npu2) {
  aiex.cert.page {
    aiex.cert.job(0) {
      aiex.cert.wait_tcts(0, 0, 1)
    }
    aiex.cert.job(1) {
      // expected-error@+1 {{more than one job on this cert.page waits on tile 0 channel 0 in uC 0}}
      aiex.cert.wait_tcts(0, 0, 2)
    }
  }
}

// -----

// Negative: same, inside an attach_to_group -- the uC is reported from the
// enclosing group.
aie.device(npu2) {
  aiex.cert.attach_to_group(2) {
    aiex.cert.page {
      aiex.cert.job(0) {
        aiex.cert.wait_tcts(0, 0, 1)
      }
      aiex.cert.job(1) {
        // expected-error@+1 {{more than one job on this cert.page waits on tile 0 channel 0 in uC 2}}
        aiex.cert.wait_tcts(0, 0, 2)
      }
    }
  }
}
