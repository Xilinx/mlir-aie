//===- insert_core_queue_pushes.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// An out-of-order channel takes the objects its queue was pushed for and
// stops, so a fifo that names no iter_count needs its core to push as it
// goes. The core counts the objects it frees and pushes once per run of as
// many objects as one task's repeat count holds.

// RUN: aie-opt --split-input-file --aie-objectfifo-split --aie-assign-packet-ids --aie-objectfifo-allocate --aie-objectfifo-insert-core-queue-pushes %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-objectfifo-split --aie-assign-packet-ids --aie-objectfifo-allocate --aie-objectfifo-insert-core-queue-pushes=check-queue-space=false %s | FileCheck %s --check-prefix=NOCHECK

// The count starts at one, so the first release pushes the run that follows
// the one the configuration pushed, and a push covers 256 objects.
// CHECK-LABEL: @runs_on
// CHECK:       aie.core(%{{.*}}) {
// CHECK:         %[[COUNT:.*]] = memref.alloca() : memref<i32>
// CHECK:         %[[ONE:.*]] = arith.constant 1 : i32
// CHECK:         memref.store %[[ONE]], %[[COUNT]][]
// CHECK:         aie.objectfifo.release @of_cons(1)
// CHECK:         %[[LEFT:.*]] = memref.load %[[COUNT]][]
// CHECK:         %[[REST:.*]] = arith.subi %[[LEFT]], %{{.*}}
// CHECK:         %[[SPENT:.*]] = arith.cmpi sle, %[[REST]], %{{.*}}
// CHECK:         scf.if %[[SPENT]] {
// CHECK:           aie.dma_poll_for_queue_space(S2MM : 0)
// CHECK:           aie.dma_push_queue(S2MM : 0) bd_id 0 {out_of_order, repeat_count = 255 : i32}
// CHECK:         }
// CHECK:         %[[RENEWED:.*]] = arith.addi %[[REST]], %{{.*}}
// CHECK:         %[[NEXT:.*]] = arith.select %[[SPENT]], %[[RENEWED]], %[[REST]]
// CHECK:         memref.store %[[NEXT]], %[[COUNT]][]

// Without the check, the push stands on its own.
// NOCHECK-LABEL: @runs_on
// NOCHECK-NOT:   aie.dma_poll_for_queue_space
// NOCHECK:       aie.dma_push_queue(S2MM : 0) bd_id 0 {out_of_order, repeat_count = 255 : i32}
module @runs_on {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_5 = aie.tile(0, 5)
    aie.objectfifo @of(%tile_0_2, {%tile_0_5}, 2 : i32) { transport = #aie.transport<dma, packet = #aie.packet_info<>>, out_of_order} : !aie.objectfifo<memref<16xi32>>
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4 step %c1 {
        %1 = aie.objectfifo.acquire @of(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @of(Produce, 1)
      }
      aie.end
    }
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4 step %c1 {
        %1 = aie.objectfifo.acquire @of(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @of(Consume, 1)
      }
      aie.end
    }
  }
}

// -----

// A fifo that names how many passes it receives is pushed for all of them at
// once, so its core pushes nothing.
// CHECK-LABEL: @bounded
// CHECK-NOT:   aie.dma_push_queue
module @bounded {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_5 = aie.tile(0, 5)
    aie.objectfifo @of(%tile_0_2, {%tile_0_5}, 2 : i32) { transport = #aie.transport<dma, packet = #aie.packet_info<>>, out_of_order, iter_count = 4 : i32} : !aie.objectfifo<memref<16xi32>>
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4 step %c1 {
        %1 = aie.objectfifo.acquire @of(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @of(Consume, 1)
      }
      aie.end
    }
  }
}

// -----

// A channel in order comes round on its own chain, so nothing pushes for it.
// CHECK-LABEL: @in_order
// CHECK-NOT:   aie.dma_push_queue
module @in_order {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_5 = aie.tile(0, 5)
    aie.objectfifo @of(%tile_0_2, {%tile_0_5}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4 step %c1 {
        %1 = aie.objectfifo.acquire @of(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @of(Consume, 1)
      }
      aie.end
    }
  }
}

// -----

// Two fifos sharing a channel share the count: every object the core frees on
// that channel comes out of the same queue.
// CHECK-LABEL: @shared_channel
// CHECK:       aie.core(%{{.*}}) {
// CHECK:         memref.alloca() : memref<i32>
// CHECK-NOT:     memref.alloca() : memref<i32>
// CHECK:         aie.end
module @shared_channel {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_5 = aie.tile(0, 5)
    aie.objectfifo @of0(%tile_0_2, {%tile_0_5}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 3>>, out_of_order, cons_ports = [#aie.end_port<DMA : 1>]} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of1(%tile_0_3, {%tile_0_5}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 4>>, out_of_order, cons_ports = [#aie.end_port<DMA : 1>]} : !aie.objectfifo<memref<16xi32>>
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4 step %c1 {
        %0 = aie.objectfifo.acquire @of0(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @of0(Consume, 1)
        %1 = aie.objectfifo.acquire @of1(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @of1(Consume, 1)
      }
      aie.end
    }
  }
}
