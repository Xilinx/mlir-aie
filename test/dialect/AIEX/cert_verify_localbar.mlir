//===- cert_verify_localbar.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// G-localbar: all cert.local_barrier ops sharing a local_barrier_id within a uC
// must be on the same cert.page.

// RUN: aie-opt --aie-cert-verify --split-input-file --verify-diagnostics %s

// Positive: both participants of local_barrier(0) are in the same page -> OK.
aie.device(xcve3858) {
  aiex.cert.page {
    aiex.cert.job(0) {
      aiex.cert.local_barrier(0, 2)
    }
    aiex.cert.job(1) {
      aiex.cert.local_barrier(0, 2)
    }
  }
}

// -----

// Positive: same id in different pages but different uCs -> OK (per-uC scope).
aie.device(xcve3858) {
  aiex.cert.attach_to_group(0) {
    aiex.cert.page {
      aiex.cert.job(0) {
        aiex.cert.local_barrier(0, 1)
      }
    }
  }
  aiex.cert.attach_to_group(2) {
    aiex.cert.page {
      aiex.cert.job(1) {
        aiex.cert.local_barrier(0, 1)
      }
    }
  }
}

// -----

// Negative: local_barrier(0) split across two pages in the same uC -> error.
aie.device(xcve3858) {
  aiex.cert.page {
    aiex.cert.job(0) {
      aiex.cert.local_barrier(0, 2)
    }
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      // expected-error@+1 {{cert.local_barrier with local_barrier_id 0 in uC 0 must be on the same page}}
      aiex.cert.local_barrier(0, 2)
    }
  }
}

// -----

// Negative: same id in two different standalone jobs (no page) is NOT
// co-located -> error. Previously "no page" was treated as a shared page for
// both, so this incorrectly passed.
aie.device(xcve3858) {
  aiex.cert.job(0) {
    aiex.cert.local_barrier(0, 2)
  }
  aiex.cert.job(1) {
    // expected-error@+1 {{cert.local_barrier with local_barrier_id 0 in uC 0 must be on the same page as its other participants}}
    aiex.cert.local_barrier(0, 2)
  }
}
