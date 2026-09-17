//===- auto-packetize-single-leg.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --aie-auto-packetize-control-ingress | FileCheck %s

// Narrowing check: when a column has no free shim MM2S channel for control, the
// pass packetizes exactly ONE ingress leg (the control trunk), never more.
//
// Three circuit shim-ingress objectfifos on column 0 (2 MM2S channels): control
// has no free channel, so one leg must be packetized. Only the trailing leg by
// symbol name (@tin2, equal traffic) is packetized and pinned to the union trunk
// K=0; @tin0 and @tin1 stay circuit and are pinned off K to channel 1. A logic
// that packetized a second leg would put `{packet}` on the @tin1 line -- the
// second CHECK-NOT window below fails under that behavior. The single packetize
// is deterministic (least-disruptive traffic, tie-broken to the trailing name).
// Re-running the pass is a no-op: @tin2 is already the packet trunk, so no
// further leg is packetized.
// The one packetized leg (@tin2) is single-hop shim->core, so its packet segment
// is confined to the shim hop via a synthesized memtile relay (@tin2_relay +
// aie.objectfifo.link); the core reads the circuit relay.
// CHECK-LABEL: aie.device(npu2) @three_input
// CHECK: %[[MEM:.*]] = aie.tile(0, 1)
// CHECK: aie.objectfifo @tin0{{.*}}prod_dma_channel = 1 : i32
// CHECK-NOT: {packet}
// CHECK: aie.objectfifo @tin1{{.*}}prod_dma_channel = 1 : i32
// CHECK-NOT: {packet}
// CHECK: aie.objectfifo @tin2(%{{.*}}, {%[[MEM]]}, {{.*}}) {packet, prod_dma_channel = 0 : i32}
// CHECK: aie.objectfifo @tin2_relay(%[[MEM]], {%{{.*}}}, {{.*}}) : !aie.objectfifo
// CHECK: aie.objectfifo.link [@tin2] -> [@tin2_relay]
// CHECK: aie.objectfifo.acquire @tin2_relay
module {
  aie.device(npu2) @three_input {
    %tshim = aie.tile(0, 0)
    %tcompute = aie.tile(0, 2)

    aie.objectfifo @tin0(%tshim, {%tcompute}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
    aie.objectfifo @tin1(%tshim, {%tcompute}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
    aie.objectfifo @tin2(%tshim, {%tcompute}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
    aie.objectfifo @tout(%tcompute, {%tshim}, 1 : i32) : !aie.objectfifo<memref<4xi32>>

    aie.core(%tcompute) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 4 : index
      %cmax = arith.constant 0xFFFFFE : index
      scf.for %niter = %c0 to %cmax step %c1 {
        %ein0 = aie.objectfifo.acquire @tin0(Consume, 1) : memref<4xi32>
        %ein1 = aie.objectfifo.acquire @tin1(Consume, 1) : memref<4xi32>
        %ein2 = aie.objectfifo.acquire @tin2(Consume, 1) : memref<4xi32>
        %eout = aie.objectfifo.acquire @tout(Produce, 1) : memref<4xi32>
        scf.for %ii = %c0 to %cn step %c1 {
          %v0 = memref.load %ein0[%ii] : memref<4xi32>
          %v1 = memref.load %ein1[%ii] : memref<4xi32>
          %v2 = memref.load %ein2[%ii] : memref<4xi32>
          %s0 = arith.addi %v0, %v1 : i32
          %r = arith.addi %s0, %v2 : i32
          memref.store %r, %eout[%ii] : memref<4xi32>
        }
        aie.objectfifo.release @tin0(Consume, 1)
        aie.objectfifo.release @tin1(Consume, 1)
        aie.objectfifo.release @tin2(Consume, 1)
        aie.objectfifo.release @tout(Produce, 1)
      }
      aie.end
    }

    aie.runtime_sequence @sequence(%a0 : memref<4xi32>, %a1 : memref<4xi32>, %a2 : memref<4xi32>, %ao : memref<4xi32>) {
      %t_in0 = aiex.dma_configure_task_for @tin0 {
        aie.dma_bd(%a0 : memref<4xi32> offset = 0 len = 4)
        aie.end
      }
      %t_in1 = aiex.dma_configure_task_for @tin1 {
        aie.dma_bd(%a1 : memref<4xi32> offset = 0 len = 4)
        aie.end
      }
      %t_in2 = aiex.dma_configure_task_for @tin2 {
        aie.dma_bd(%a2 : memref<4xi32> offset = 0 len = 4)
        aie.end
      }
      %t_out = aiex.dma_configure_task_for @tout {
        aie.dma_bd(%ao : memref<4xi32> offset = 0 len = 4)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_in0)
      aiex.dma_start_task(%t_in1)
      aiex.dma_start_task(%t_in2)
      aiex.dma_start_task(%t_out)
      aiex.dma_await_task(%t_out)
      aiex.dma_free_task(%t_in0)
      aiex.dma_free_task(%t_in1)
      aiex.dma_free_task(%t_in2)
      aiex.dma_free_task(%t_out)
    }
  }
}
