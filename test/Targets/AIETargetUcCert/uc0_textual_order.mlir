//===- uc0_textual_order.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The emitter must preserve textual (encounter) order within a uC stream across
// device-level content and attach_to_group content. Here a release job authored
// in attach_to_group(0) appears BEFORE a device-level runtime-sequence page
// (containing WAIT_TCTS); it must emit first on uC 0. The previous two-phase
// emitter emitted all device-level pages before any attach_to_group content,
// putting WAIT_TCTS on an earlier page than the release -- a forward-page
// dependency, which is what mis-ordered real multi-uC designs.

// RUN: aie-translate --aie-cert-to-asm %s | FileCheck %s

// CHECK: .attach_to_group 0
// CHECK: START_JOB 0
// CHECK: WRITE_32{{.*}}0x007ae050
// CHECK: START_JOB 2
// CHECK: WAIT_TCTS

aie.device(npu2) {
  // Release job, authored first, on uC 0.
  aiex.cert.attach_to_group(0) {
    aiex.cert.job(0) {
      aiex.cert.write32(0x7AE050, 1)
    }
  }
  // Runtime-sequence page (waits for TCTs), authored second, also uC 0.
  aiex.cert.page {
    aiex.cert.job(2) {
      aiex.cert.wait_tcts(0, 0, 1)
    }
  }
}
