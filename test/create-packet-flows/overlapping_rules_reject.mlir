//===- overlapping_rules_reject.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-verify-packet-rules %s

// The switchbox from Xilinx/mlir-aie#437. A switch matches rules in order and
// routes on the first hit, so an id matching both goes West by rule 1 and the
// South rule never sees it. Ids 10, 11, 14 and 15 are shared; the packet_flow
// ops make 10 and 14 live, and the diagnostic names the lowest live one.
module {
  aie.device(xcvc1902) {
    %tile = aie.tile(2, 3)
    aie.packet_flow(10) { aie.packet_source<%tile, DMA : 0>  aie.packet_dest<%tile, West : 0> }
    aie.packet_flow(14) { aie.packet_source<%tile, DMA : 0>  aie.packet_dest<%tile, South : 0> }
    %sb = aie.switchbox(%tile) {
      %west = aie.amsel<0> (0)
      %south = aie.amsel<0> (1)
      %0 = aie.masterset(South : 0, %south)
      %1 = aie.masterset(West : 0, %west)
      aie.packet_rules(DMA : 0) {
        // expected-remark@+1 {{this is the rule that claims packet id 10}}
        aie.rule(26, 10, %west)
        // expected-error@+1 {{'aie.rule' op is shadowed for packet id 10}}
        aie.rule(24, 8, %south)
      }
    }
  }
}

// -----

// Relaxed masks that overlap only on ids nothing sends are how the router fits
// a port's flows into its slots -- id 18 is claimed by both rules and by no
// flow, which is what --aie-create-pathfinder-flows emits for live ids
// {10, 22} -> North:1 and {16, 19} -> North:4.
module {
  aie.device(npu1_1col) {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    aie.packet_flow(10) { aie.packet_source<%t00, DMA : 0>  aie.packet_dest<%t01, DMA : 0> }
    aie.packet_flow(22) { aie.packet_source<%t00, DMA : 0>  aie.packet_dest<%t01, DMA : 0> }
    aie.packet_flow(16) { aie.packet_source<%t00, DMA : 0>  aie.packet_dest<%t01, DMA : 1> }
    aie.packet_flow(19) { aie.packet_source<%t00, DMA : 0>  aie.packet_dest<%t01, DMA : 1> }
    %sb = aie.switchbox(%t00) {
      %up = aie.amsel<0> (0)
      %mem = aie.amsel<1> (0)
      %0 = aie.masterset(North : 1, %up)
      %1 = aie.masterset(North : 4, %mem)
      aie.packet_rules(South : 3) {
        aie.rule(3, 2, %up)
        aie.rule(28, 16, %mem)
      }
    }
  }
}

// -----

// Exact masks cannot overlap at all.
module {
  aie.device(xcvc1902) {
    %tile = aie.tile(2, 3)
    aie.packet_flow(10) { aie.packet_source<%tile, DMA : 0>  aie.packet_dest<%tile, West : 0> }
    aie.packet_flow(14) { aie.packet_source<%tile, DMA : 0>  aie.packet_dest<%tile, South : 0> }
    %sb = aie.switchbox(%tile) {
      %west = aie.amsel<0> (0)
      %south = aie.amsel<0> (1)
      %0 = aie.masterset(South : 0, %south)
      %1 = aie.masterset(West : 0, %west)
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 10, %west)
        aie.rule(31, 14, %south)
      }
    }
  }
}

// -----

// Rules naming the same amsel take the same route, so an overlap between them
// is redundant rather than a misroute. Here two distinct AMSelOps spell the
// same (arbiter, msel).
module {
  aie.device(xcvc1902) {
    %tile = aie.tile(2, 3)
    aie.packet_flow(10) { aie.packet_source<%tile, DMA : 0>  aie.packet_dest<%tile, West : 0> }
    %sb = aie.switchbox(%tile) {
      %a = aie.amsel<0> (0)
      %b = aie.amsel<0> (0)
      %0 = aie.masterset(West : 0, %a)
      %1 = aie.masterset(South : 0, %b)
      aie.packet_rules(DMA : 0) {
        aie.rule(30, 10, %a)
        aie.rule(31, 10, %b)
      }
    }
  }
}

// -----

// A rule whose value has bits outside its mask matches nothing: (id & 24) is
// one of {0, 8, 16, 24} and never 10, so it cannot shadow the rule after it.
module {
  aie.device(xcvc1902) {
    %tile = aie.tile(2, 3)
    aie.packet_flow(8) { aie.packet_source<%tile, DMA : 0>  aie.packet_dest<%tile, South : 0> }
    %sb = aie.switchbox(%tile) {
      %west = aie.amsel<0> (0)
      %south = aie.amsel<0> (1)
      %0 = aie.masterset(South : 0, %south)
      %1 = aie.masterset(West : 0, %west)
      aie.packet_rules(DMA : 0) {
        aie.rule(24, 10, %west)
        aie.rule(24, 8, %south)
      }
    }
  }
}
