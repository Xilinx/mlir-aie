//===- test_sa_stream_port_no_allocate.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-place-tiles='placer=sa_placer sa-seed=1' %s | FileCheck %s

// A fifo whose consumer end is a core stream port moves by DMA wherever its
// ends sit, so the placer neither counts it as shared memory between the two
// neighbouring cores nor asks for an aie.objectfifo.allocate, which the
// verifier would refuse.
// CHECK: aie.objectfifo @s
// CHECK-NOT: aie.objectfifo.allocate
module {
  aie.device(npu2) {
    %p = aie.logical_tile<CoreTile>(0, 2)
    %c = aie.logical_tile<CoreTile>(0, 3)
    %big = aie.buffer(%p) : memref<8000xi32>
    aie.objectfifo @s (%p, {%c}, 2 : i32) {cons_ports = [#aie.end_port<Core : 0>]} : !aie.objectfifo<memref<16xi32>>
  }
}
