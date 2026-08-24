//===- multi_uc_capstone.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// End-to-end capstone for the CERT ordering/placement model on a target with
// more than one microcontroller, covering the shape a real multi-uC design
// takes:
//  - @configure is the first page on uC0;
//  - uC0 releases its core, rendezvous with col1 via a remote barrier, then
//    waits for TCTs -- all in one page (release before WAIT_TCTS, no forward
//    dependency);
//  - col1's controller (uC1, via placement=1) rendezvous on the same remote
//    barrier and releases its core;
//  - the two uCs form independent ordered streams;
//  - aie-cert-verify accepts it (matching rendezvous, valid ids).

// The design verifies cleanly (no diagnostics => aie-opt exits 0).
// RUN: aie-opt -cert-legalize-pages -aie-cert-verify %s | FileCheck %s

// Config page is first on uC0; col1 work is grouped under uC 1.
// CHECK: aiex.cert.job(1)
// CHECK: aiex.cert.write32(393216, 1)
// CHECK: aiex.cert.attach_to_group(1)

// RUN: aie-opt -cert-legalize-pages %s | aie-translate --aie-cert-to-asm | FileCheck %s --check-prefix=ASM

// uC0 stream: config first, then release-before-WAIT_TCTS in one page.
// ASM: .attach_to_group 0
// ASM: START_JOB 1
// ASM: WRITE_32{{.*}}0x00060000
// ASM: START_JOB 2
// ASM: WRITE_32{{.*}}0x007ae050
// ASM-NOT: .eop
// ASM: REMOTE_BARRIER
// ASM: WAIT_TCTS
// uC1 stream: rendezvous then release col1's core.
// ASM: .attach_to_group 1
// ASM: REMOTE_BARRIER
// ASM: WRITE_32{{.*}}0x026ae050

aie.device(xcve3858) {
  // Configuration job (forced first on uC0).
  aiex.cert.job(1) {
    aiex.cert.write32(0x60000, 1)
  } {cert.configure}

  // uC0: release col0 core, rendezvous with col1, wait for TCTs (one page).
  aiex.cert.page {
    aiex.cert.job(2) {
      aiex.cert.write32(0x7AE050, 1)
      aiex.cert.remote_barrier(1, 0x3)
      aiex.cert.wait_tcts(0, 0, 1)
    }
  }

  // uC1 (col1 controller): rendezvous, then release col1 core.
  aiex.cert.page {
    aiex.cert.job(3) {
      aiex.cert.remote_barrier(1, 0x3)
      aiex.cert.write32(0x26AE050, 1)
    }
  } {placement = 1 : i32}
}
