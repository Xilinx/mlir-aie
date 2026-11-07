//===- resource_stages_multicast.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --pass-pipeline='builtin.module(aie-resource-stages{load-pdi-to-ctrl-pkt=true})' %s | FileCheck %s
// RUN: aie-opt --pass-pipeline='builtin.module(aie-resource-stages{load-pdi-to-ctrl-pkt=true ctrlpkt-multicast-span=0})' %s | FileCheck %s --check-prefix=OFF
// RUN: aie-opt --pass-pipeline='builtin.module(aie-resource-stages{generate-ctrl-pkt-overlay=true})' %s | FileCheck %s --check-prefix=OFF

// As in aiecc, the two core tiles get a control multicast under
// load-pdi-to-ctrl-pkt, by default, and none with the span at 0 or with an
// overlay alone.

// CHECK: aie.ctrl_pkt_multicast = [
// OFF-NOT: aie.ctrl_pkt_multicast

module {
  aie.device(npu1_1col) {
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    %a = aie.logical_tile<CoreTile>(?, ?)
    %b = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @in(%shim, {%a}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @mid(%a, {%b}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%a) {
      %i = aie.objectfifo.acquire @in (Consume, 1) : memref<16xi32>
      %o = aie.objectfifo.acquire @mid (Produce, 1) : memref<16xi32>
      aie.objectfifo.release @in (Consume, 1)
      aie.objectfifo.release @mid (Produce, 1)
      aie.end
    }
    aie.core(%b) {
      %i = aie.objectfifo.acquire @mid (Consume, 1) : memref<16xi32>
      aie.objectfifo.release @mid (Consume, 1)
      aie.end
    }
  }
}
