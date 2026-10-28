//===- contraction-multiplicity.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Contraction combines equivalent buffer paths and records their multiplicity.

// REQUIRES: aie-visualize
// RUN: aie-visualize --emit-dot %s | FileCheck %s --check-prefix=DEFAULT
// RUN: aie-visualize --emit-dot --contract=port,buffer %s | FileCheck %s
// RUN: aie-visualize --emit-dot --contract=port,dma,buffer --contract-balanced-fanout %s | FileCheck %s --check-prefix=ALL
// RUN: aie-visualize --emit-dot --contract-balanced-fanout %s | FileCheck %s --check-prefix=BALANCED

// CHECK: call_0 {{.*}}label="consume"
// CHECK-DAG: p_0_3_1_0_s -> buffer_0
// CHECK-DAG: p_0_3_1_0_s -> buffer_1
// CHECK-DAG: p_0_3_1_1_s -> buffer_0
// CHECK-DAG: p_0_3_1_1_s -> buffer_1
// CHECK-DAG: buffer_0 -> call_0
// CHECK-DAG: buffer_1 -> call_0

// ALL-NOT: buffer_0 [
// ALL-NOT: buffer_1 [
// ALL-DAG: p_0_2_0_0_in_r0 -> call_0 {{.*}}label={{.*}}&#215;4

// DEFAULT-DAG: p_0_3_1_0_s -> buffer_0
// DEFAULT-DAG: p_0_3_1_0_s -> buffer_1
// DEFAULT-DAG: buffer_0 -> call_0
// DEFAULT-DAG: buffer_1 -> call_0

// BALANCED-NOT: buffer_0 [
// BALANCED-NOT: buffer_1 [
// BALANCED-DAG: buffer_2 -> call_1
// BALANCED-DAG: buffer_3 -> call_2
// BALANCED-DAG: p_0_3_1_0_s -> call_0 {{.*}}label={{.*}}&#215;2
// BALANCED-DAG: p_0_3_1_1_s -> call_0 {{.*}}label={{.*}}&#215;2
// BALANCED-DAG: p_1_3_1_0_s -> buffer_2
// BALANCED-DAG: p_1_3_1_0_s -> buffer_3

module {
  aie.device(npu1_2col) {
    func.func private @consume(memref<16xi32>, memref<16xi32>)
    func.func private @other(memref<16xi32>)
    func.func private @third(memref<16xi32>)

    %source = aie.tile(0, 2)
    %sink = aie.tile(0, 3)
    %other_sink = aie.tile(1, 3)
    %buffer0 = aie.buffer(%sink) : memref<16xi32>
    %buffer1 = aie.buffer(%sink) : memref<16xi32>
    %buffer2 = aie.buffer(%other_sink) : memref<16xi32>
    %buffer3 = aie.buffer(%other_sink) : memref<16xi32>

    aie.core(%sink) {
      func.call @consume(%buffer0, %buffer1) : (memref<16xi32>, memref<16xi32>) -> ()
      aie.end
    }

    aie.core(%other_sink) {
      func.call @other(%buffer2) : (memref<16xi32>) -> ()
      func.call @third(%buffer3) : (memref<16xi32>) -> ()
      aie.end
    }

    aie.mem(%sink) {
      %0 = aie.dma_start(S2MM, 0, ^bd0, ^ch1)
    ^bd0:
      aie.dma_bd(%buffer0 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%buffer1 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^bd0
    ^ch1:
      %1 = aie.dma_start(S2MM, 1, ^shared0, ^end)
    ^shared0:
      aie.dma_bd(%buffer0 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^shared1
    ^shared1:
      aie.dma_bd(%buffer1 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^shared0
    ^end:
      aie.end
    }

    aie.mem(%other_sink) {
      %0 = aie.dma_start(S2MM, 0, ^other, ^end)
    ^other:
      aie.dma_bd(%buffer2 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^third
    ^third:
      aie.dma_bd(%buffer3 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^other
    ^end:
      aie.end
    }

    aie.flow(%source, Core : 0, %sink, DMA : 0)
      via (%source : Core : 0 -> North : 0,
           %sink : South : 0 -> DMA : 0)
    aie.flow(%source, Core : 0, %sink, DMA : 1)
      via (%source : Core : 0 -> North : 1,
           %sink : South : 1 -> DMA : 1)
    aie.flow(%source, Core : 0, %other_sink, DMA : 0)
      via (%source : Core : 0 -> North : 2,
           %other_sink : West : 2 -> DMA : 0)
  }
}
