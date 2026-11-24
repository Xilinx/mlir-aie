//===- test_place_pinned_core_neighbours.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Two core-to-core fifos one core sends on pinned MM2S 0 clash only where
// both go by DMA: beside their consumers, shared memory carries them and the
// pin holds nothing. Both placers place the design, and it lowers with at
// most one of them on DMA. (A pair that moves by DMA wherever it goes still
// fails: test_place_pinned_channel_unshared.mlir.)

// RUN: aie-opt --verify-diagnostics --pass-pipeline='builtin.module(aie-resource-stages)' %s | FileCheck %s
// RUN: aie-opt --verify-diagnostics --pass-pipeline='builtin.module(aie-resource-stages{placer=sa_placer sa-seed=1})' %s | FileCheck %s
// RUN: aie-opt --verify-diagnostics --pass-pipeline='builtin.module(aie-resource-stages{placer=sa_placer sa-seed=2})' %s | FileCheck %s

// CHECK-LABEL: module @core_neighbours
// CHECK-COUNT-3: aie.core(

module @core_neighbours {
  aie.device(npu2_1col) {
    %a = aie.logical_tile<CoreTile>(?, ?)
    %b = aie.logical_tile<CoreTile>(?, ?)
    %c = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @x(%a, {%b}, 2 : i32) {prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @y(%a, {%c}, 2 : i32) {prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.core(%a) {
      %0 = aie.objectfifo.acquire @x(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @x(Produce, 1)
      %1 = aie.objectfifo.acquire @y(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @y(Produce, 1)
      aie.end
    }
    aie.core(%b) {
      %0 = aie.objectfifo.acquire @x(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @x(Consume, 1)
      aie.end
    }
    aie.core(%c) {
      %0 = aie.objectfifo.acquire @y(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @y(Consume, 1)
      aie.end
    }
  }
}
