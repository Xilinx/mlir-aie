//===- cert_register_ops_invalid.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Register operands are r0..r23; a write32_d field is a register index only
// when flagged.

// RUN: aie-opt --split-input-file --verify-diagnostics %s

aie.device(xcve3858) {
  aiex.cert.page {
    aiex.cert.job(1) {
      // expected-error@+1 {{attribute 'value' failed to satisfy constraint}}
      aiex.cert.read32(24, 0x1000)
    }
  }
}

// -----

aie.device(xcve3858) {
  aiex.cert.page {
    aiex.cert.job(1) {
      // expected-error@+1 {{attribute 'address' failed to satisfy constraint}}
      aiex.cert.read32_d(24, 1)
    }
  }
}

// -----

aie.device(xcve3858) {
  aiex.cert.page {
    aiex.cert.job(1) {
      // expected-error@+1 {{has address_is_reg, so address 4096 must be a register index in [0, 23]}}
      aiex.cert.write32_d(0x1000, 5) {address_is_reg}
    }
  }
}

// -----

aie.device(xcve3858) {
  aiex.cert.page {
    aiex.cert.job(1) {
      // expected-error@+1 {{has value_is_reg, so value 24 must be a register index in [0, 23]}}
      aiex.cert.write32_d(0x1000, 24) {value_is_reg}
    }
  }
}
