//===- cert_verify_job_slots.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Job IDs on one page must differ modulo 64, the CERT firmware's job slots.

// RUN: aie-opt -aie-cert-verify --split-input-file --verify-diagnostics %s

aie.device(xcve3858) {
  aiex.cert.page {
    aiex.cert.job(1) {}
    // expected-error@+1 {{job ID 65 collides with job ID 1 modulo 64 on the same page}}
    aiex.cert.job(65) {}
  }
}

// -----

aie.device(xcve3858) {
  aiex.cert.page {
    aiex.cert.job(0) {}
    // expected-error@+1 {{job ID 0 collides with job ID 0 modulo 64 on the same page}}
    aiex.cert.job(0) {}
  }
}

// -----

// Slots can be reused across pages, uCs, and sections.
aie.device(xcve3858) {
  aiex.cert.page { aiex.cert.job(1) {} }
  aiex.cert.page { aiex.cert.job(65) {} }
  aiex.cert.attach_to_group(1) {
    aiex.cert.page { aiex.cert.job(1) {} }
  }
  aiex.cert.section @cfg {
    aiex.cert.page { aiex.cert.job(65) {} }
  }
}
