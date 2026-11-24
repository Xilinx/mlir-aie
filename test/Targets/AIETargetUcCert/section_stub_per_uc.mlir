//===- section_stub_per_uc.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// AIEBU resolves labels within each controller's block, so every group that
// loads a section gets the section's stub after its header. A section loaded
// from inside another section is loaded from every group that loads that one.
// Block 0 keeps every stub, and a group's stubs are emitted once.

// RUN: rm -rf section_stub_per_uc && mkdir section_stub_per_uc
// RUN: aie-translate --aie-cert-to-asm --cert-section-dir=section_stub_per_uc %s | FileCheck %s

// CHECK-LABEL: .attach_to_group 0
// CHECK-EMPTY:
// CHECK-NEXT:  inner:
// CHECK-NEXT:  .include inner.asm
// CHECK-NEXT:  .endl inner
// CHECK-EMPTY:
// CHECK-NEXT:  outer:
// CHECK-NEXT:  .include outer.asm
// CHECK-NEXT:  .endl outer
// CHECK-EMPTY:
// CHECK-NEXT:  config:
// CHECK-NEXT:  .include config.asm
// CHECK-NEXT:  .endl config
// CHECK-EMPTY:
// CHECK-NEXT:  extra:
// CHECK-NEXT:  .include extra.asm
// CHECK-NEXT:  .endl extra
// CHECK-EMPTY:
// CHECK-NEXT:  .align 16
// CHECK:       LOAD_PDI               1, @config
// CHECK:       LOAD_PDI               2, @outer
// CHECK:       EOF

// @outer loads @inner, so group 1 needs both.
// CHECK-LABEL: .attach_to_group 1
// CHECK-EMPTY:
// CHECK-NEXT:  inner:
// CHECK-NEXT:  .include inner.asm
// CHECK-NEXT:  .endl inner
// CHECK-EMPTY:
// CHECK-NEXT:  outer:
// CHECK-NEXT:  .include outer.asm
// CHECK-NEXT:  .endl outer
// CHECK-EMPTY:
// CHECK-NEXT:  config:
// CHECK-NEXT:  .include config.asm
// CHECK-NEXT:  .endl config
// CHECK-EMPTY:
// CHECK-NEXT:  .align 16
// CHECK:       LOAD_PDI               2, @outer
// CHECK:       LOAD_PDI               1, @config
// CHECK:       EOF

// CHECK-LABEL: .attach_to_group 2
// CHECK-EMPTY:
// CHECK-NEXT:  inner:
// CHECK-NEXT:  .include inner.asm
// CHECK-NEXT:  .endl inner
// CHECK-EMPTY:
// CHECK-NEXT:  config:
// CHECK-NEXT:  .include config.asm
// CHECK-NEXT:  .endl config
// CHECK-EMPTY:
// CHECK-NEXT:  extra:
// CHECK-NEXT:  .include extra.asm
// CHECK-NEXT:  .endl extra
// CHECK-EMPTY:
// CHECK-NEXT:  .align 16
// CHECK:       LOAD_PDI               1, @config
// CHECK:       LOAD_PDI               3, @inner
// CHECK:       .attach_to_group 2
// CHECK-EMPTY:
// CHECK-NEXT:  .align 16
// CHECK:       LOAD_PDI               4, @extra
// CHECK:       EOF

module {
  aie.device(npu2) {
    aiex.cert.section @inner {
      aiex.cert.page {
        aiex.cert.job(1) {
          aiex.cert.write32(0x1000, 1)
        }
      }
    }
    aiex.cert.section @outer {
      aiex.cert.page {
        aiex.cert.job(1) {
          aiex.cert.load_pdi(3, @inner)
        }
      }
    }
    aiex.cert.section @config {
      aiex.cert.page {
        aiex.cert.job(1) {
          aiex.cert.write32(0x2000, 2)
        }
      }
    }
    aiex.cert.section @extra {
      aiex.cert.page {
        aiex.cert.job(1) {
          aiex.cert.write32(0x3000, 3)
        }
      }
    }
    aiex.cert.page {
      aiex.cert.job(2) {
        aiex.cert.load_pdi(1, @config)
      }
    }
    aiex.cert.page {
      aiex.cert.job(3) {
        aiex.cert.load_pdi(2, @outer)
      }
    }
    aiex.cert.attach_to_group(1) {
      aiex.cert.page {
        aiex.cert.job(2) {
          aiex.cert.load_pdi(2, @outer)
        }
      }
      aiex.cert.page {
        aiex.cert.job(3) {
          aiex.cert.load_pdi(1, @config)
        }
      }
    }
    aiex.cert.attach_to_group(2) {
      aiex.cert.page {
        aiex.cert.job(2) {
          aiex.cert.load_pdi(1, @config)
        }
      }
      aiex.cert.page {
        aiex.cert.job(3) {
          aiex.cert.load_pdi(3, @inner)
        }
      }
    }
    aiex.cert.attach_to_group(2) {
      aiex.cert.page {
        aiex.cert.job(4) {
          aiex.cert.load_pdi(4, @extra)
        }
      }
    }
  }
}
