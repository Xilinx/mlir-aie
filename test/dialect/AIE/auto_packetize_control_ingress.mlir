//===- auto_packetize_control_ingress.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --aie-auto-packetize-control-ingress | FileCheck %s

// Two circuit shim-ingress objectfifos on column 0 (2 MM2S channels): the column
// is saturated, so exactly one leg is packetized (the trailing one by symbol
// name) and pinned to the union trunk channel K, while the other stays circuit
// and is pinned off K. A single-input design packetizes nothing; its lone leg is
// simply pinned off K so K stays free for control. The packetized leg is
// symbol-name-deterministic (least-disruptive traffic, tie-broken to the trailing
// name), so re-running the pass on its own output is a no-op.

// @two_input: @in0 and @in1 both shim(0,0) -> compute(0,2), 2 MM2S channels on
// column 0 -> K = 0. @in1 (trailing by name, equal traffic) is packetized and
// pinned to K=0; @in0 stays circuit and is pinned to channel 1. Because @in1 is a
// single-hop shim->core leg, the pass keeps the packet segment on the contested
// shim hop only: @in1 is retargeted to a synthesized column memtile (0,1) and a
// new circuit relay @in1_relay(memtile -> compute) plus an aie.objectfifo.link
// forwards it onward, so the core consumes @in1_relay. The positive @in0 CHECK
// anchors at the symbol, so the CHECK-NOT window up to @in1 catches an erroneous
// both-flip that would put `{packet}` on @in0.
// CHECK-LABEL: aie.device(npu2) @two_input
// CHECK: %[[MEM:.*]] = aie.tile(0, 1)
// CHECK: aie.objectfifo @in0{{.*}}prod_dma_channel = 1 : i32
// CHECK-NOT: {packet}
// CHECK: aie.objectfifo @in1(%{{.*}}, {%[[MEM]]}, {{.*}}) {packet, prod_dma_channel = 0 : i32}
// CHECK: aie.objectfifo @in1_relay(%[[MEM]], {%{{.*}}}, {{.*}}) : !aie.objectfifo
// CHECK: aie.objectfifo.link [@in1] -> [@in1_relay]
// The core reads the circuit relay, not the packet shim leg.
// CHECK: aie.objectfifo.acquire @in1_relay
module {
  aie.device(npu2) @two_input {
    %tshim = aie.tile(0, 0)
    %tcompute = aie.tile(0, 2)

    aie.objectfifo @in0(%tshim, {%tcompute}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
    aie.objectfifo @in1(%tshim, {%tcompute}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
    aie.objectfifo @out(%tcompute, {%tshim}, 1 : i32) : !aie.objectfifo<memref<4xi32>>

    aie.core(%tcompute) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 4 : index
      %cmax = arith.constant 0xFFFFFE : index
      scf.for %niter = %c0 to %cmax step %c1 {
        %ein0 = aie.objectfifo.acquire @in0(Consume, 1) : memref<4xi32>
        %ein1 = aie.objectfifo.acquire @in1(Consume, 1) : memref<4xi32>
        %eout = aie.objectfifo.acquire @out(Produce, 1) : memref<4xi32>
        scf.for %ii = %c0 to %cn step %c1 {
          %v0 = memref.load %ein0[%ii] : memref<4xi32>
          %v1 = memref.load %ein1[%ii] : memref<4xi32>
          %r = arith.addi %v0, %v1 : i32
          memref.store %r, %eout[%ii] : memref<4xi32>
        }
        aie.objectfifo.release @in0(Consume, 1)
        aie.objectfifo.release @in1(Consume, 1)
        aie.objectfifo.release @out(Produce, 1)
      }
      aie.end
    }

    aie.runtime_sequence @sequence(%a0 : memref<4xi32>, %a1 : memref<4xi32>, %ao : memref<4xi32>) {
      %t_in0 = aiex.dma_configure_task_for @in0 {
        aie.dma_bd(%a0 : memref<4xi32> offset = 0 len = 4)
        aie.end
      }
      %t_in1 = aiex.dma_configure_task_for @in1 {
        aie.dma_bd(%a1 : memref<4xi32> offset = 0 len = 4)
        aie.end
      }
      %t_out = aiex.dma_configure_task_for @out {
        aie.dma_bd(%ao : memref<4xi32> offset = 0 len = 4)
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

  // @one_input: a single shim-ingress circuit objectfifo (1 of the 2 MM2S channels
  // on column 0) -> K = 0 stays free, nothing is packetized, and the lone leg is
  // pinned off K to channel 1. The CHECK-NOT window is bounded below by the @sin0
  // symbol anchor and above by the trailing @sout positive CHECK, so it genuinely
  // covers @sin0's attr region: a wrong flip of the single input would put
  // `{packet}` on the @sin0 line and fail here.
  // CHECK-LABEL: aie.device(npu2) @one_input
  // CHECK: aie.objectfifo @sin0{{.*}}prod_dma_channel = 1 : i32
  // CHECK-NOT: {packet}
  // CHECK: aie.objectfifo @sout
  aie.device(npu2) @one_input {
    %tshim = aie.tile(0, 0)
    %tcompute = aie.tile(0, 2)

    aie.objectfifo @sin0(%tshim, {%tcompute}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
    aie.objectfifo @sout(%tcompute, {%tshim}, 1 : i32) : !aie.objectfifo<memref<4xi32>>

    aie.core(%tcompute) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 4 : index
      %cmax = arith.constant 0xFFFFFE : index
      scf.for %niter = %c0 to %cmax step %c1 {
        %ein0 = aie.objectfifo.acquire @sin0(Consume, 1) : memref<4xi32>
        %eout = aie.objectfifo.acquire @sout(Produce, 1) : memref<4xi32>
        scf.for %ii = %c0 to %cn step %c1 {
          %v0 = memref.load %ein0[%ii] : memref<4xi32>
          %r = arith.addi %v0, %v0 : i32
          memref.store %r, %eout[%ii] : memref<4xi32>
        }
        aie.objectfifo.release @sin0(Consume, 1)
        aie.objectfifo.release @sout(Produce, 1)
      }
      aie.end
    }

    aie.runtime_sequence @single_sequence(%a0 : memref<4xi32>, %ao : memref<4xi32>) {
      %t_in0 = aiex.dma_configure_task_for @sin0 {
        aie.dma_bd(%a0 : memref<4xi32> offset = 0 len = 4)
        aie.end
      }
      %t_out = aiex.dma_configure_task_for @sout {
        aie.dma_bd(%ao : memref<4xi32> offset = 0 len = 4)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_in0)
      aiex.dma_start_task(%t_out)
      aiex.dma_await_task(%t_out)
      aiex.dma_free_task(%t_in0)
      aiex.dma_free_task(%t_out)
    }
  }
}
