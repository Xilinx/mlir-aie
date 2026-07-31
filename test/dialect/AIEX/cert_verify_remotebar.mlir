//===- cert_verify_remotebar.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// G-uC / rendezvous: a cert.remote_barrier must actually rendezvous. Every uC
// in the party_mask must be present AND have a matching remote_barrier with the
// same id/mask; each participant sets its own bit.

// RUN: aie-opt --aie-cert-verify --split-input-file --verify-diagnostics %s

// Positive: uCs 0 and 1 both have remote_barrier(1, 0x3); masks agree, own bits
// set -> OK.
aie.device(npu2) {
  aiex.cert.attach_to_group(0) {
    aiex.cert.job(0) {
      aiex.cert.remote_barrier(1, 0x3)
    }
  }
  aiex.cert.attach_to_group(1) {
    aiex.cert.job(1) {
      aiex.cert.remote_barrier(1, 0x3)
    }
  }
}

// -----

// Negative: mask 0x5 references uC 2, which has no group -> error.
aie.device(npu2) {
  aiex.cert.attach_to_group(0) {
    aiex.cert.job(0) {
      // expected-error@+1 {{party_mask references uC 2 which is not present in the design}}
      aiex.cert.remote_barrier(1, 0x5)
    }
  }
  aiex.cert.attach_to_group(1) {
    aiex.cert.job(1) {
      aiex.cert.write32(0x2000, 43)
    }
  }
}

// -----

// Negative (the key rendezvous check): uC 0 waits on remote_barrier(1, 0x3) but
// uC 1 has no matching barrier -> uC 0 would wait forever.
aie.device(npu2) {
  aiex.cert.attach_to_group(0) {
    aiex.cert.job(0) {
      // expected-error@+1 {{party_mask includes uC 1 but that uC has no matching remote_barrier(1) to rendezvous with}}
      aiex.cert.remote_barrier(1, 0x3)
    }
  }
  aiex.cert.attach_to_group(1) {
    aiex.cert.job(1) {
      aiex.cert.write32(0x2000, 43)
    }
  }
}

// -----

// Negative: both uCs participate, but uC 0's own bit is not set in the mask.
aie.device(npu2) {
  aiex.cert.attach_to_group(0) {
    aiex.cert.job(0) {
      // expected-error@+1 {{on uC 0 excludes its own uC from party_mask}}
      aiex.cert.remote_barrier(1, 0x2)
    }
  }
  aiex.cert.attach_to_group(1) {
    aiex.cert.job(1) {
      aiex.cert.remote_barrier(1, 0x2)
    }
  }
}
