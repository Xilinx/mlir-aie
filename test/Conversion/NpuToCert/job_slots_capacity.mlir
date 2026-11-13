//===- job_slots_capacity.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Every job ID below is a multiple of 64, so all share slot 0. A page of 64
// such jobs uses all 64 firmware slots; a page of 65 cannot be allocated.

// RUN: aie-opt -cert-legalize-pages -aie-cert-verify --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: aie.device(xcve3858)
// CHECK: aiex.cert.job(0) {
// CHECK-NEXT: }
// CHECK-NEXT: aiex.cert.job(1) {
// CHECK-COUNT-61: aiex.cert.job(
// CHECK: aiex.cert.job(63) {
// CHECK-NOT: aiex.cert.job
aie.device(xcve3858) {
  aiex.cert.page {
    aiex.cert.job(0) {} aiex.cert.job(64) {} aiex.cert.job(128) {} aiex.cert.job(192) {}
    aiex.cert.job(256) {} aiex.cert.job(320) {} aiex.cert.job(384) {} aiex.cert.job(448) {}
    aiex.cert.job(512) {} aiex.cert.job(576) {} aiex.cert.job(640) {} aiex.cert.job(704) {}
    aiex.cert.job(768) {} aiex.cert.job(832) {} aiex.cert.job(896) {} aiex.cert.job(960) {}
    aiex.cert.job(1024) {} aiex.cert.job(1088) {} aiex.cert.job(1152) {} aiex.cert.job(1216) {}
    aiex.cert.job(1280) {} aiex.cert.job(1344) {} aiex.cert.job(1408) {} aiex.cert.job(1472) {}
    aiex.cert.job(1536) {} aiex.cert.job(1600) {} aiex.cert.job(1664) {} aiex.cert.job(1728) {}
    aiex.cert.job(1792) {} aiex.cert.job(1856) {} aiex.cert.job(1920) {} aiex.cert.job(1984) {}
    aiex.cert.job(2048) {} aiex.cert.job(2112) {} aiex.cert.job(2176) {} aiex.cert.job(2240) {}
    aiex.cert.job(2304) {} aiex.cert.job(2368) {} aiex.cert.job(2432) {} aiex.cert.job(2496) {}
    aiex.cert.job(2560) {} aiex.cert.job(2624) {} aiex.cert.job(2688) {} aiex.cert.job(2752) {}
    aiex.cert.job(2816) {} aiex.cert.job(2880) {} aiex.cert.job(2944) {} aiex.cert.job(3008) {}
    aiex.cert.job(3072) {} aiex.cert.job(3136) {} aiex.cert.job(3200) {} aiex.cert.job(3264) {}
    aiex.cert.job(3328) {} aiex.cert.job(3392) {} aiex.cert.job(3456) {} aiex.cert.job(3520) {}
    aiex.cert.job(3584) {} aiex.cert.job(3648) {} aiex.cert.job(3712) {} aiex.cert.job(3776) {}
    aiex.cert.job(3840) {} aiex.cert.job(3904) {} aiex.cert.job(3968) {} aiex.cert.job(4032) {}
  }
}

// -----

aie.device(xcve3858) {
  // expected-error@+1 {{cert.page has 65 jobs, exceeding the 64 firmware job slots}}
  aiex.cert.page {
    aiex.cert.job(0) {} aiex.cert.job(64) {} aiex.cert.job(128) {} aiex.cert.job(192) {}
    aiex.cert.job(256) {} aiex.cert.job(320) {} aiex.cert.job(384) {} aiex.cert.job(448) {}
    aiex.cert.job(512) {} aiex.cert.job(576) {} aiex.cert.job(640) {} aiex.cert.job(704) {}
    aiex.cert.job(768) {} aiex.cert.job(832) {} aiex.cert.job(896) {} aiex.cert.job(960) {}
    aiex.cert.job(1024) {} aiex.cert.job(1088) {} aiex.cert.job(1152) {} aiex.cert.job(1216) {}
    aiex.cert.job(1280) {} aiex.cert.job(1344) {} aiex.cert.job(1408) {} aiex.cert.job(1472) {}
    aiex.cert.job(1536) {} aiex.cert.job(1600) {} aiex.cert.job(1664) {} aiex.cert.job(1728) {}
    aiex.cert.job(1792) {} aiex.cert.job(1856) {} aiex.cert.job(1920) {} aiex.cert.job(1984) {}
    aiex.cert.job(2048) {} aiex.cert.job(2112) {} aiex.cert.job(2176) {} aiex.cert.job(2240) {}
    aiex.cert.job(2304) {} aiex.cert.job(2368) {} aiex.cert.job(2432) {} aiex.cert.job(2496) {}
    aiex.cert.job(2560) {} aiex.cert.job(2624) {} aiex.cert.job(2688) {} aiex.cert.job(2752) {}
    aiex.cert.job(2816) {} aiex.cert.job(2880) {} aiex.cert.job(2944) {} aiex.cert.job(3008) {}
    aiex.cert.job(3072) {} aiex.cert.job(3136) {} aiex.cert.job(3200) {} aiex.cert.job(3264) {}
    aiex.cert.job(3328) {} aiex.cert.job(3392) {} aiex.cert.job(3456) {} aiex.cert.job(3520) {}
    aiex.cert.job(3584) {} aiex.cert.job(3648) {} aiex.cert.job(3712) {} aiex.cert.job(3776) {}
    aiex.cert.job(3840) {} aiex.cert.job(3904) {} aiex.cert.job(3968) {} aiex.cert.job(4032) {}
    aiex.cert.job(4096) {}
  }
}
