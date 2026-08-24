//===- uc_stream_separation.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// a non-zero group must be emitted as its own uC stream (its own EOF),
// even when there is device-level (group 0) content and no explicit
// attach_to_group(0). Previously the emitter keyed blocks by loop counter, so a
// lone non-zero group was lumped into the group-0 block and shared its EOF.

// RUN: aie-translate --aie-cert-to-asm %s | FileCheck %s

// CHECK: .attach_to_group 0
// CHECK: START_JOB 0
// CHECK: EOF
// CHECK: .attach_to_group 2
// CHECK: START_JOB 1
// CHECK: EOF

module {
  aie.device(xcve3858) {
    // device-level page => uC 0
    aiex.cert.page {
      aiex.cert.job(0) {
        aiex.cert.write32(0x1000, 1)
      }
    }
    // placed on uC 2 => its own stream
    aiex.cert.attach_to_group(2) {
      aiex.cert.page {
        aiex.cert.job(1) {
          aiex.cert.write32(0x2000, 2)
        }
      }
    }
  }
}
