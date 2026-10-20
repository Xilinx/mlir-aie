//===- contraction-multiplicity.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Contraction combines equivalent buffer paths and records their multiplicity.

// REQUIRES: aie-visualize
// RUN: aie-visualize --emit-dot --contract=port,buffer %s | FileCheck %s

// CHECK-NOT: buffer_
// CHECK: call_0 {{.*}}label="consume"
// CHECK-NOT: buffer_
// The two buffers contract away; their multiplicity is carried by the DMA
// channel both of them reach, which is where they genuinely meet.
// CHECK-DAG: tile_0_2 -> p_0_3_1_0_s
// CHECK-DAG: p_0_3_1_0_s -> call_0 {{.*}}label={{.*}}&#215;2
// CHECK-NOT: buffer_

module {
  aie.device(npu1_1col) {
    func.func private @consume(memref<16xi32>, memref<16xi32>)

    %source = aie.tile(0, 2)
    %sink = aie.tile(0, 3)
    %buffer0 = aie.buffer(%sink) : memref<16xi32>
    %buffer1 = aie.buffer(%sink) : memref<16xi32>

    aie.core(%sink) {
      func.call @consume(%buffer0, %buffer1) : (memref<16xi32>, memref<16xi32>) -> ()
      aie.end
    }

    aie.mem(%sink) {
      %0 = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%buffer0 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%buffer1 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }

    aie.flow(%source, Core : 0, %sink, DMA : 0)
      via (%source : Core : 0 -> North : 0,
           %sink : South : 0 -> DMA : 0)
  }
}
