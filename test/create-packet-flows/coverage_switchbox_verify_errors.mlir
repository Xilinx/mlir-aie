//===- coverage_switchbox_verify_errors.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %sb02 = aie.switchbox(%t02) {
      aie.connect<DMA : 0, North : 0>
      // expected-error@+1 {{targets same dst as another connect op; existing destinations: (North: 0)}}
      aie.connect<DMA : 1, North : 0>
    }
  }
}

// -----

module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %sb02 = aie.switchbox(%t02) {
      // expected-error@+1 {{index 9 for dest bundle North must be less than 6}}
      aie.connect<DMA : 0, North : 9>
    }
  }
}

// -----

module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %sb02 = aie.switchbox(%t02) {
      // expected-error@+1 {{dest index cannot be less than zero}}
      aie.connect<DMA : 0, North : -1>
    }
  }
}

// -----

module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %sb02 = aie.switchbox(%t02) {
      %a0 = aie.amsel<0> (0)
      // expected-error@+1 {{index 9 for dest bundle North must be less than 6}}
      aie.masterset(North : 9, %a0)
    }
  }
}

// -----

module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %sb02 = aie.switchbox(%t02) {
      %a0 = aie.amsel<0> (0)
      // expected-error@+1 {{dest index cannot be less than zero}}
      aie.masterset(North : -1, %a0)
    }
  }
}

// -----

module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %sb02 = aie.switchbox(%t02) {
      // expected-error@+1 {{cannot be contained in a Switchbox op}}
      %c = arith.constant 0 : index
    }
  }
}

// -----

module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    aie.packet_flow(1) {
      aie.packet_source<%t02, DMA : 0>
      aie.packet_dest<%t03, DMA : 0>
      // expected-error@+1 {{cannot be contained in a PacketFlow op}}
      %c = arith.constant 0 : index
    }
  }
}
