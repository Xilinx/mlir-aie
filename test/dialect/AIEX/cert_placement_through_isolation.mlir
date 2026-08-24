//===- cert_placement_through_isolation.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Regression for the placement-loss bug: a placed page (placement = 2) whose
// job contains a preempt is split by full-page isolation into multiple pages.
// Because placement is lowered to attach_to_group BEFORE isolation, every
// resulting page stays on uC 2 (previously isolation created unplaced pages
// that silently emitted on group 0).

// RUN: aie-opt -cert-legalize-pages %s | FileCheck %s

// CHECK: aiex.cert.attach_to_group(2)
// CHECK: aiex.cert.write32(4096, 42)
// CHECK: aiex.cert.preempt(0, @save, @restore)
// CHECK-NOT: aiex.cert.attach_to_group

module {
  aie.device(xcve3858) {
    aiex.cert.section @save {
      aiex.cert.page {
        aiex.cert.job(10) {
          aiex.cert.write32(0x2100000, 0)
        }
      }
    }
    aiex.cert.section @restore {
      aiex.cert.page {
        aiex.cert.job(11) {
          aiex.cert.write32(0x2100000, 1)
        }
      }
    }
    aiex.cert.page {
      aiex.cert.job(0) {
        aiex.cert.write32(0x1000, 42)
        aiex.cert.preempt(0, @save, @restore)
      }
    } {placement = 2 : i32}
  }
}
