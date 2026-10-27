//===- invalid_sparse_flow_vias.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-split-flow-vias %s

module {
  aie.device(xcvc1902) {
    %src = aie.tile(0, 1)
    %via = aie.tile(0, 8)
    // expected-error@+1 {{via ingress has no neighboring tile}}
    aie.flow(%src, DMA : 0, %via, DMA : 0) via (%via : North : 0 -> DMA : 0)
  }
}

// -----

module {
  aie.device(npu2_3col) {
    %src = aie.tile(1, 1)
    %via = aie.tile(0, 2)
    %dst = aie.tile(1, 3)
    // expected-error@+1 {{via ingress has no neighboring tile}}
    aie.packet_flow(1) {
      aie.packet_source<%src, DMA : 0>
      aie.packet_dest<%dst, DMA : 0>
    } via (%via : West : 0 -> East : 0)
  }
}

// -----

module {
  aie.device(xcvc1902) {
    %src = aie.tile(0, 7)
    %via = aie.tile(0, 8)
    %dst = aie.tile(0, 1)
    // expected-error@+1 {{via egress has no neighboring tile}}
    aie.flow(%src, DMA : 0, %dst, DMA : 0) via (%via : South : 0 -> North : 0)
  }
}
