//===- bad_cert_section_parent.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// Section names are device symbols, and each section is written to a file
// named after it, so a section is defined at device level. A load on any
// controller names it from there.

aie.device(xcve3858) {
  aiex.cert.attach_to_group(1) {
    // expected-error @+1 {{'aiex.cert.section' op expects parent op 'aie.device'}}
    aiex.cert.section @config {
      aiex.cert.page {
        aiex.cert.job(1) {
          aiex.cert.write32(0x1000, 1)
        }
      }
    }
  }
}

// -----

aie.device(xcve3858) {
  aiex.cert.section @config {
    aiex.cert.page {
      aiex.cert.job(1) {
        aiex.cert.write32(0x1000, 1)
      }
    }
  }
  aiex.cert.attach_to_group(1) {
    aiex.cert.page {
      aiex.cert.job(1) {
        aiex.cert.load_pdi(1, @config)
      }
    }
  }
}
