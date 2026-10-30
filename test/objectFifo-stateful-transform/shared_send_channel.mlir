//===- shared_send_channel.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A sending channel that cycles on one fifo's buffers never reaches another's,
// so two fifos can only share it when the core pushes the descriptor of each
// object it releases. That is release_strategy = push_queue, which pinning two
// fifos to one sending channel selects.

// RUN: aie-opt --split-input-file --aie-objectFifo-stateful-transform --aie-assign-lock-ids --aie-assign-buffer-addresses --aie-assign-bd-ids %s | FileCheck %s

// Both fifos send over channel 0, and the configuration pushes neither: each
// release pushes the descriptor of the buffer it frees.
// CHECK-LABEL: @shared_send
// CHECK:       aie.core(%{{.*}}) {
// CHECK:         aie.dma_push_queue(MM2S : 0) bd_id 0
// CHECK:         aie.dma_push_queue(MM2S : 0) bd_id 1
// CHECK:         aie.dma_push_queue(MM2S : 0) bd_id 2
// CHECK:         aie.dma_push_queue(MM2S : 0) bd_id 3
// CHECK:       aie.mem(%{{.*}}) {
// CHECK:         aie.dma_start(MM2S, 0, {{.*}}) {core_pushes}
// CHECK:         aie.dma_bd(%a_buff_0 {{.*}}) {bd_id = 0 : i32
// CHECK:         aie.dma_bd(%a_buff_1 {{.*}}) {bd_id = 1 : i32
// CHECK:         aie.dma_start(MM2S, 0, {{.*}}) {core_pushes}
// CHECK:         aie.dma_bd(%b_buff_0 {{.*}}) {bd_id = 2 : i32
// CHECK:         aie.dma_bd(%b_buff_1 {{.*}}) {bd_id = 3 : i32
module @shared_send {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)
    aie.objectfifo @a(%tile_0_2, {%tile_0_4}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 3>>, prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%tile_0_2, {%tile_0_5}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 4>>, prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4 step %c1 {
        %0 = aie.objectfifo.acquire @a(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @a(Produce, 1)
        %1 = aie.objectfifo.acquire @b(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @b(Produce, 1)
      }
      aie.end
    }
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4 step %c1 {
        %0 = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @a(Consume, 1)
      }
      aie.end
    }
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4 step %c1 {
        %0 = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @b(Consume, 1)
      }
      aie.end
    }
  }
}

// -----

// The strategy stands on its own: a fifo with a channel to itself still has
// its core push, when asked.
// CHECK-LABEL: @alone
// CHECK:       aie.dma_push_queue(MM2S : 0) bd_id 0
// CHECK:       aie.dma_start(MM2S, 0, {{.*}}) {core_pushes}
module @alone {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_4 = aie.tile(0, 4)
    aie.objectfifo @a(%tile_0_2, {%tile_0_4}, 2 : i32) {release_strategy = #aie.release_strategy<push_queue>} : !aie.objectfifo<memref<16xi32>>
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4 step %c1 {
        %0 = aie.objectfifo.acquire @a(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @a(Produce, 1)
      }
      aie.end
    }
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4 step %c1 {
        %0 = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @a(Consume, 1)
      }
      aie.end
    }
  }
}

// -----

// Without the strategy, the sending DMA cycles on the buffers and the core
// pushes nothing.
// CHECK-LABEL: @cycles
// CHECK-NOT:   aie.dma_push_queue
// CHECK-NOT:   core_pushes
module @cycles {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_4 = aie.tile(0, 4)
    aie.objectfifo @a(%tile_0_2, {%tile_0_4}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4 step %c1 {
        %0 = aie.objectfifo.acquire @a(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @a(Produce, 1)
      }
      aie.end
    }
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4 step %c1 {
        %0 = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @a(Consume, 1)
      }
      aie.end
    }
  }
}
