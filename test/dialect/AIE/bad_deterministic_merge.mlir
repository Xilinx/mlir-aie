//===- bad_deterministic_merge.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file -verify-diagnostics

// Deterministic merge is scoped to a whole arbiter, so a slave port routed to
// that arbiter with no slot of its own would never be granted again once the
// feature is enabled -- a silent, permanent stall. Reject it.
module {
  aie.device(npu1) {
    %t = aie.tile(0, 3)
    aie.switchbox(%t) {
      // expected-error @+1 {{does not schedule slave port DMA : 0}}
      %a0 = aie.amsel<0> (0) deterministic_merge [ <North : 1, 1>, <North : 1, 2>]
      aie.masterset(South : 1, %a0)
      aie.packet_rules(North : 1) { aie.rule(31, 2, %a0) }
      aie.packet_rules(DMA : 0) { aie.rule(31, 1, %a0) }
    }
  }
}

// -----

// Several amsels may name one arbiter, but the schedule covers the arbiter, so
// only one of them may carry it.
module {
  aie.device(npu1) {
    %t = aie.tile(0, 3)
    aie.switchbox(%t) {
      // expected-error @+1 {{already has a deterministic merge schedule on another amsel}}
      %a0 = aie.amsel<0> (0) deterministic_merge [ <North : 1, 1>, <DMA : 0, 1>]
      %a1 = aie.amsel<0> (1) deterministic_merge [ <North : 1, 1>, <DMA : 0, 1>]
      aie.masterset(South : 1, %a0, %a1)
      aie.packet_rules(North : 1) { aie.rule(31, 2, %a0) }
      aie.packet_rules(DMA : 0) { aie.rule(31, 1, %a0) }
    }
  }
}

// -----

// Only arbiters 0 and 1 implement the feature on AIE2.
module {
  aie.device(npu1) {
    %t = aie.tile(0, 3)
    aie.switchbox(%t) {
      // expected-error @+1 {{arbiter 2 does not support deterministic merge}}
      %a0 = aie.amsel<2> (0) deterministic_merge [ <North : 1, 1>, <DMA : 0, 1>]
      aie.masterset(South : 1, %a0)
      aie.packet_rules(North : 1) { aie.rule(31, 2, %a0) }
      aie.packet_rules(DMA : 0) { aie.rule(31, 1, %a0) }
    }
  }
}

// -----

// The hardware marks an unused slot with a packet count of zero and slots 0 and
// 1 may never be zero, so the narrowest legal configuration is 2-to-1.
module {
  aie.device(npu1) {
    %t = aie.tile(0, 3)
    aie.switchbox(%t) {
      // expected-error @+1 {{needs at least 2 merge slots}}
      %a0 = aie.amsel<0> (0) deterministic_merge [ <North : 1, 1>]
      aie.masterset(South : 1, %a0)
      aie.packet_rules(North : 1) { aie.rule(31, 2, %a0) }
    }
  }
}

// -----

// packet_count is a 6-bit field.
module {
  aie.device(npu1) {
    %t = aie.tile(0, 3)
    aie.switchbox(%t) {
      // expected-error @+1 {{packet_count must be in 1..63}}
      %a0 = aie.amsel<0> (0) deterministic_merge [ <North : 1, 64>, <DMA : 0, 1>]
      aie.masterset(South : 1, %a0)
      aie.packet_rules(North : 1) { aie.rule(31, 2, %a0) }
      aie.packet_rules(DMA : 0) { aie.rule(31, 1, %a0) }
    }
  }
}

// -----

// AIE1 stream switches have no such feature at all.
module {
  aie.device(xcvc1902) {
    %t = aie.tile(2, 3)
    aie.switchbox(%t) {
      // expected-error @+1 {{deterministic merge is not supported by this device}}
      %a0 = aie.amsel<0> (0) deterministic_merge [ <North : 1, 1>, <DMA : 0, 1>]
      aie.masterset(South : 1, %a0)
      aie.packet_rules(North : 1) { aie.rule(31, 2, %a0) }
      aie.packet_rules(DMA : 0) { aie.rule(31, 1, %a0) }
    }
  }
}
