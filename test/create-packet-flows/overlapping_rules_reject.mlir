//===- overlapping_rules_reject.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// The switchbox from Xilinx/mlir-aie#437. A switch matches rules in order and
// routes on the first hit, so ids matched by both go West by rule 1 and the
// South rule is dead for them -- id 14 (0b01110) is the one the issue reports:
// 14 & 26 == 10 and 14 & 24 == 8. Ids 10, 11 and 15 are shared too; the
// diagnostic names the lowest.
module {
  aie.device(xcvc1902) {
    %tile = aie.tile(2, 3)
    %sb = aie.switchbox(%tile) {
      %west = aie.amsel<0> (0)
      %south = aie.amsel<0> (1)
      %0 = aie.masterset(South : 0, %south)
      %1 = aie.masterset(West : 0, %west)
      aie.packet_rules(DMA : 0) {
        aie.rule(26, 10, %west)
        // expected-error@+1 {{'aie.rule' op is shadowed for packet id 10}}
        aie.rule(24, 8, %south)
      }
    }
  }
}

// -----

// Exact masks cannot overlap. This is what --aie-create-pathfinder-flows emits
// for the same two ids.
module {
  aie.device(xcvc1902) {
    %tile = aie.tile(2, 3)
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
// is redundant rather than a misroute, and stays legal.
module {
  aie.device(xcvc1902) {
    %tile = aie.tile(2, 3)
    %sb = aie.switchbox(%tile) {
      %west = aie.amsel<0> (0)
      %0 = aie.masterset(West : 0, %west)
      aie.packet_rules(DMA : 0) {
        aie.rule(30, 10, %west)
        aie.rule(31, 10, %west)
      }
    }
  }
}

// -----

// Same route spelled with two distinct AMSelOps for the same (arbiter, msel).
module {
  aie.device(xcvc1902) {
    %tile = aie.tile(2, 3)
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
// one of {0, 8, 16, 24} and never 10. It cannot shadow the rule after it.
module {
  aie.device(xcvc1902) {
    %tile = aie.tile(2, 3)
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
