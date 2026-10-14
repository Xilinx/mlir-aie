//===- bad_parent_core_flow.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Cores and flows belong directly to a device (or, before canonicalize-device,
// to the module): the device verifier, the router and the CDO read them there.

// RUN: aie-opt --split-input-file --verify-diagnostics %s

aie.device(npu2) {
  %t = aie.tile(0, 2)
  scf.execute_region {
    // expected-error@+1 {{'aie.core' op expects parent op to be one of 'aie.device, builtin.module'}}
    %core = aie.core(%t) {
      aie.end
    }
    scf.yield
  }
}

// -----

aie.device(npu2) {
  %a = aie.tile(0, 2)
  %b = aie.tile(0, 3)
  scf.execute_region {
    // expected-error@+1 {{'aie.flow' op expects parent op to be one of 'aie.device, builtin.module'}}
    aie.flow(%a, DMA : 0, %b, DMA : 0)
    scf.yield
  }
}

// -----

aie.device(npu2) {
  %a = aie.tile(0, 2)
  %b = aie.tile(0, 3)
  scf.execute_region {
    // expected-error@+1 {{'aie.packet_flow' op expects parent op to be one of 'aie.device, builtin.module'}}
    aie.packet_flow(1) {
      aie.packet_source<%a, DMA : 0>
      aie.packet_dest<%b, DMA : 0>
    }
    scf.yield
  }
}
