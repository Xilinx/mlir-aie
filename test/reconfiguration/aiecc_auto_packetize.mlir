// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: rm -rf %t && mkdir -p %t/adaptive %t/fixed
// RUN: cd %t/adaptive && %aiecc --load-pdi-to-ctrl-pkt --get=input_physical.mlir %s
// RUN: FileCheck %s < %t/adaptive/input_physical.mlir
// RUN: cd %t/fixed && not %aiecc --load-pdi-to-ctrl-pkt --ctrlpkt-adaptive-overlay=false --get=input_physical.mlir %s 2>&1 | FileCheck %s --check-prefix=FIXED

// CHECK: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan1
// FIXED: Unable to find a legal routing

module {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %comp = aie.tile(0, 2)
    %t3 = aie.tile(0, 3)
    %t4 = aie.tile(0, 4)
    %t5 = aie.tile(0, 5)
    aie.objectfifo @in0 (%shim, {%comp}, 1 : i32) {transport = #aie.transport<auto, packet = #aie.packet_info<>>} : !aie.objectfifo<memref<4xi32>>
    aie.objectfifo @in1 (%shim, {%comp}, 1 : i32) {prod_port = #aie.end_port<DMA : 0>, transport = #aie.transport<auto, packet = #aie.packet_info<>>} : !aie.objectfifo<memref<4xi32>>
    aie.objectfifo @out (%comp, {%shim}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
    aie.core(%comp) {
      %e0 = aie.objectfifo.acquire @in0 (Consume, 1) : memref<4xi32>
      %e1 = aie.objectfifo.acquire @in1 (Consume, 1) : memref<4xi32>
      %eo = aie.objectfifo.acquire @out (Produce, 1) : memref<4xi32>
      aie.objectfifo.release @in0 (Consume, 1)
      aie.objectfifo.release @in1 (Consume, 1)
      aie.objectfifo.release @out (Produce, 1)
      aie.end
    }
    aie.packet_flow(2) { aie.packet_source<%shim, DMA : 0> aie.packet_dest<%mem, DMA : 1> }
    aie.packet_flow(4) { aie.packet_source<%shim, DMA : 0> aie.packet_dest<%mem, DMA : 2> }
    aie.packet_flow(8) { aie.packet_source<%shim, DMA : 0> aie.packet_dest<%mem, DMA : 3> }
    aie.packet_flow(16) { aie.packet_source<%shim, DMA : 0> aie.packet_dest<%mem, DMA : 4> }
    aie.packet_flow(5) { aie.packet_source<%shim, DMA : 0> aie.packet_dest<%mem, DMA : 5> }
    aie.packet_flow(12) { aie.packet_source<%shim, DMA : 0> aie.packet_dest<%t3, DMA : 0> }
    aie.packet_flow(20) { aie.packet_source<%shim, DMA : 0> aie.packet_dest<%t4, DMA : 0> }
    aie.packet_flow(21) { aie.packet_source<%shim, DMA : 0> aie.packet_dest<%t5, DMA : 0> }
    aie.packet_flow(22) { aie.packet_source<%shim, DMA : 0> aie.packet_dest<%t3, DMA : 1> }
    aie.packet_flow(23) { aie.packet_source<%shim, DMA : 0> aie.packet_dest<%t4, DMA : 1> }
    aie.packet_flow(24) { aie.packet_source<%shim, DMA : 0> aie.packet_dest<%t5, DMA : 1> }
    aie.runtime_sequence @sequence(%a0 : memref<4xi32>, %a1 : memref<4xi32>, %argout : memref<4xi32>) {
      %t_in0 = aiex.dma_configure_task_for @in0 {
        aie.dma_bd(%a0 : memref<4xi32> offset = 0 len = 4)
        aie.end
      }
      %t_in1 = aiex.dma_configure_task_for @in1 {
        aie.dma_bd(%a1 : memref<4xi32> offset = 0 len = 4)
        aie.end
      }
      %t_out = aiex.dma_configure_task_for @out {
        aie.dma_bd(%argout : memref<4xi32> offset = 0 len = 4)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_in0)
      aiex.dma_start_task(%t_in1)
      aiex.dma_start_task(%t_out)
      aiex.dma_await_task(%t_out)
      aiex.dma_free_task(%t_in0)
      aiex.dma_free_task(%t_in1)
      aiex.dma_free_task(%t_out)
    }
  }
}
