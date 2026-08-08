//===- routing_errors.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Error paths of --aie-create-pathfinder-flows. These pin diagnostics that had
// no test coverage.

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-create-pathfinder-flows %s

// A packet id wider than the target's id field cannot be encoded in a rule.
module {
  aie.device(npu1_1col) {
    // The diagnostic is emitted at the tile whose switchbox needed the rule.
    // expected-error@+1 {{packet id 32 exceeds the maximum of 31}}
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    aie.packet_flow(32) { aie.packet_source<%t00, DMA : 0>  aie.packet_dest<%t01, DMA : 0> }
  }
}

// -----

// Three sources fanning in to shim S2MM 1 while the shim also sources and sinks
// other flows: the arbiter assignment reaches a master/slave pairing the shim
// stream switch cannot express. A router limitation rather than an invalid
// design -- pinned so a change in behaviour is noticed.
module {
  aie.device(npu1_1col) {
    // expected-error@+1 {{'aie.amsel' op illegal stream switch connection}}
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t05 = aie.tile(0, 5)
    aie.packet_flow(13) { aie.packet_source<%t02, DMA : 0>  aie.packet_dest<%t01, DMA : 2> }
    aie.packet_flow(31) { aie.packet_source<%t03, DMA : 1>  aie.packet_dest<%t00, DMA : 0> }
    aie.packet_flow(28) { aie.packet_source<%t03, DMA : 0>  aie.packet_dest<%t00, DMA : 1> }
    aie.packet_flow(6) { aie.packet_source<%t05, DMA : 0>  aie.packet_dest<%t00, DMA : 1> }
    aie.packet_flow(2) { aie.packet_source<%t00, DMA : 1>  aie.packet_dest<%t01, DMA : 1> }
    aie.packet_flow(27) { aie.packet_source<%t02, DMA : 1>  aie.packet_dest<%t00, DMA : 1> }
  }
}

// -----

// Multicast flows out of one source port with overlapping destination sets each
// need their own msel on the master port's arbiter, and an arbiter has four.
module {
  aie.device(npu1_1col) {
    %t00 = aie.tile(0, 0)
    // expected-error@+1 {{'aie.tile' op tile op arbiter 1 has used up all its msels}}
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    aie.packet_flow(8) {
      aie.packet_source<%t00, DMA : 1>
      aie.packet_dest<%t02, DMA : 0>
    }
    aie.packet_flow(1) {
      aie.packet_source<%t00, DMA : 1>
      aie.packet_dest<%t01, DMA : 0>
      aie.packet_dest<%t03, DMA : 0>
    }
    aie.packet_flow(24) {
      aie.packet_source<%t00, DMA : 1>
      aie.packet_dest<%t01, DMA : 0>
    }
    aie.packet_flow(20) {
      aie.packet_source<%t00, DMA : 1>
      aie.packet_dest<%t02, DMA : 0>
    }
    aie.packet_flow(17) {
      aie.packet_source<%t00, DMA : 0>
      aie.packet_dest<%t04, DMA : 0>
      aie.packet_dest<%t01, DMA : 0>
      aie.packet_dest<%t01, DMA : 2>
    }
  }
}
