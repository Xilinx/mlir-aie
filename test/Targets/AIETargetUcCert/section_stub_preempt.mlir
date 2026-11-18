//===- section_stub_preempt.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// PREEMPT's save/restore labels are resolved within each controller's block,
// like LOAD_PDI's. A group that reaches a PREEMPT, directly or through a loaded
// section, gets stubs for its save/restore sections and the sections they load.

// RUN: rm -rf section_stub_preempt && mkdir section_stub_preempt
// RUN: aie-translate --aie-cert-to-asm --cert-section-dir=section_stub_preempt %s | FileCheck %s

// CHECK-LABEL: .attach_to_group 0
// CHECK-EMPTY:
// CHECK-NEXT:  inner:
// CHECK:       save:
// CHECK:       restore:
// CHECK:       outer:
// CHECK:       .align 16
// CHECK:       PREEMPT                0x0000, @save, @restore
// CHECK:       EOF

// Group 1 loads @outer, whose PREEMPT names @save and @restore; @restore
// loads @inner.
// CHECK-LABEL: .attach_to_group 1
// CHECK-EMPTY:
// CHECK-NEXT:  inner:
// CHECK-NEXT:  .include inner.asm
// CHECK-NEXT:  .endl inner
// CHECK-EMPTY:
// CHECK-NEXT:  save:
// CHECK-NEXT:  .include save.asm
// CHECK-NEXT:  .endl save
// CHECK-EMPTY:
// CHECK-NEXT:  restore:
// CHECK-NEXT:  .include restore.asm
// CHECK-NEXT:  .endl restore
// CHECK-EMPTY:
// CHECK-NEXT:  outer:
// CHECK-NEXT:  .include outer.asm
// CHECK-NEXT:  .endl outer
// CHECK-EMPTY:
// CHECK-NEXT:  .align 16
// CHECK:       LOAD_PDI               1, @outer
// CHECK:       EOF

// Group 2's PREEMPT names @save and @restore directly; it does not need @outer.
// CHECK-LABEL: .attach_to_group 2
// CHECK-EMPTY:
// CHECK-NEXT:  inner:
// CHECK-NEXT:  .include inner.asm
// CHECK-NEXT:  .endl inner
// CHECK-EMPTY:
// CHECK-NEXT:  save:
// CHECK-NEXT:  .include save.asm
// CHECK-NEXT:  .endl save
// CHECK-EMPTY:
// CHECK-NEXT:  restore:
// CHECK-NEXT:  .include restore.asm
// CHECK-NEXT:  .endl restore
// CHECK-EMPTY:
// CHECK-NEXT:  .align 16
// CHECK:       PREEMPT                0x0000, @save, @restore
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
    aiex.cert.section @save {
      aiex.cert.page {
        aiex.cert.job(1) {
          aiex.cert.write32(0x2000, 2)
        }
      }
    }
    aiex.cert.section @restore {
      aiex.cert.page {
        aiex.cert.job(1) {
          aiex.cert.load_pdi(2, @inner)
        }
      }
    }
    aiex.cert.section @outer {
      aiex.cert.page {
        aiex.cert.job(1) {
          aiex.cert.preempt(0, @save, @restore)
        }
      }
    }
    aiex.cert.page {
      aiex.cert.job(2) {
        aiex.cert.preempt(0, @save, @restore)
      }
    }
    aiex.cert.attach_to_group(1) {
      aiex.cert.page {
        aiex.cert.job(2) {
          aiex.cert.load_pdi(1, @outer)
        }
      }
    }
    aiex.cert.attach_to_group(2) {
      aiex.cert.page {
        aiex.cert.job(2) {
          aiex.cert.preempt(0, @save, @restore)
        }
      }
    }
  }
}
