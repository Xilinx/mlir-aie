// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: aie-opt --split-input-file --aie-objectFifo-stateful-transform --aie-assign-bd-ids %s | FileCheck %s
//
// Queue capacity belongs to a physical tile and MM2S channel. Each sending
// pool contributes its depth once, regardless of how many releases use it.
// Acquiring two objects gives the corresponding sending pool depth three.
// CHECK-LABEL: @shared_exceeds
module @shared_exceeds {
  aie.device(npu2) {
    %src = aie.tile(0, 2)
    %dst_a = aie.tile(0, 4)
    %dst_b = aie.tile(1, 4)
    aie.objectfifo @a(%src, {%dst_a}, [3, 2]) {packet, packet_id = 3 : i8, prod_dma_channel = 0 : i32, release_strategy = #aie.release_strategy<push_queue>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%src, {%dst_b}, [2, 2]) {packet, packet_id = 4 : i8, prod_dma_channel = 0 : i32, release_strategy = #aie.release_strategy<push_queue>} : !aie.objectfifo<memref<16xi32>>
    %core0 = aie.core(%src) {
      // CHECK: aie.dma_poll_for_queue_space(MM2S : 0)
      // CHECK-NEXT: aie.dma_push_queue(MM2S : 0) bd_id 0
      // CHECK: scf.index_switch
      // CHECK: aie.dma_poll_for_queue_space(MM2S : 0)
      // CHECK-NEXT: aie.dma_push_queue(MM2S : 0) bd_id 3
      %c0 = arith.constant 0 : index
      %c6 = arith.constant 6 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c6 step %c1 {
        %a0, %a1 = aie.objectfifo.acquire @a(Produce, 2) : memref<16xi32>, memref<16xi32>
        aie.objectfifo.release @a(Produce, 1)
        %b0 = aie.objectfifo.acquire @b(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @b(Produce, 1)
      }
      aie.end
    }
    %recv_a = aie.core(%dst_a) {
      %c0 = arith.constant 0 : index
      %c6 = arith.constant 6 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c6 step %c1 {
        %a0 = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @a(Consume, 1)
      }
      aie.end
    }
    %recv_b = aie.core(%dst_b) {
      %c0 = arith.constant 0 : index
      %c6 = arith.constant 6 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c6 step %c1 {
        %b0 = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @b(Consume, 1)
      }
      aie.end
    }
  }
}

// -----

// CHECK-LABEL: @shared_fits
// CHECK-NOT: aie.dma_poll_for_queue_space
// CHECK: aie.dma_push_queue(MM2S : 0)
// CHECK-NOT: aie.dma_poll_for_queue_space
// CHECK: aie.dma_push_queue(MM2S : 0)
// CHECK-NOT: aie.dma_poll_for_queue_space
module @shared_fits {
  aie.device(npu2) {
    %src = aie.tile(0, 2)
    %dst_a = aie.tile(0, 4)
    %dst_b = aie.tile(1, 4)
    aie.objectfifo @a(%src, {%dst_a}, [2, 2]) {packet, packet_id = 3 : i8, prod_dma_channel = 0 : i32, release_strategy = #aie.release_strategy<push_queue>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%src, {%dst_b}, [2, 2]) {packet, packet_id = 4 : i8, prod_dma_channel = 0 : i32, release_strategy = #aie.release_strategy<push_queue>} : !aie.objectfifo<memref<16xi32>>
    %core0 = aie.core(%src) {
      %c0 = arith.constant 0 : index
      %c6 = arith.constant 6 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c6 step %c1 {
        %a0 = aie.objectfifo.acquire @a(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @a(Produce, 1)
        %b0 = aie.objectfifo.acquire @b(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @b(Produce, 1)
      }
      aie.end
    }
    %recv_a = aie.core(%dst_a) {
      %c0 = arith.constant 0 : index
      %c6 = arith.constant 6 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c6 step %c1 {
        %a0 = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @a(Consume, 1)
      }
      aie.end
    }
    %recv_b = aie.core(%dst_b) {
      %c0 = arith.constant 0 : index
      %c6 = arith.constant 6 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c6 step %c1 {
        %b0 = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @b(Consume, 1)
      }
      aie.end
    }
  }
}

// -----

// CHECK-LABEL: @separate_channels
// CHECK-NOT: aie.dma_poll_for_queue_space
// CHECK: aie.dma_push_queue(MM2S : 0)
// CHECK-NOT: aie.dma_poll_for_queue_space
// CHECK: aie.dma_push_queue(MM2S : 1)
// CHECK-NOT: aie.dma_poll_for_queue_space
module @separate_channels {
  aie.device(npu2) {
    %src = aie.tile(0, 2)
    %dst_a = aie.tile(0, 4)
    %dst_b = aie.tile(1, 4)
    aie.objectfifo @a(%src, {%dst_a}, [3, 2]) {packet, packet_id = 3 : i8, prod_dma_channel = 0 : i32, release_strategy = #aie.release_strategy<push_queue>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%src, {%dst_b}, [3, 2]) {packet, packet_id = 4 : i8, prod_dma_channel = 1 : i32, release_strategy = #aie.release_strategy<push_queue>} : !aie.objectfifo<memref<16xi32>>
    %core0 = aie.core(%src) {
      %c0 = arith.constant 0 : index
      %c6 = arith.constant 6 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c6 step %c1 {
        %a0, %a1 = aie.objectfifo.acquire @a(Produce, 2) : memref<16xi32>, memref<16xi32>
        aie.objectfifo.release @a(Produce, 1)
        %b0, %b1 = aie.objectfifo.acquire @b(Produce, 2) : memref<16xi32>, memref<16xi32>
        aie.objectfifo.release @b(Produce, 1)
      }
      aie.end
    }
    %recv_a = aie.core(%dst_a) {
      %c0 = arith.constant 0 : index
      %c6 = arith.constant 6 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c6 step %c1 {
        %a0 = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @a(Consume, 1)
      }
      aie.end
    }
    %recv_b = aie.core(%dst_b) {
      %c0 = arith.constant 0 : index
      %c6 = arith.constant 6 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c6 step %c1 {
        %b0 = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @b(Consume, 1)
      }
      aie.end
    }
  }
}

// -----

// CHECK-LABEL: @separate_tiles
// CHECK-NOT: aie.dma_poll_for_queue_space
// CHECK: aie.dma_push_queue(MM2S : 0)
// CHECK-NOT: aie.dma_poll_for_queue_space
// CHECK: aie.core
// CHECK-NOT: aie.dma_poll_for_queue_space
// CHECK: aie.dma_push_queue(MM2S : 0)
// CHECK-NOT: aie.dma_poll_for_queue_space
module @separate_tiles {
  aie.device(npu2) {
    %src = aie.tile(0, 2)
    %other = aie.tile(1, 2)
    %dst_a = aie.tile(0, 4)
    %dst_b = aie.tile(1, 4)
    aie.objectfifo @a(%src, {%dst_a}, [3, 2]) {packet, packet_id = 3 : i8, prod_dma_channel = 0 : i32, release_strategy = #aie.release_strategy<push_queue>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%other, {%dst_b}, [3, 2]) {packet, packet_id = 4 : i8, prod_dma_channel = 0 : i32, release_strategy = #aie.release_strategy<push_queue>} : !aie.objectfifo<memref<16xi32>>
    %core0 = aie.core(%src) {
      %c0 = arith.constant 0 : index
      %c6 = arith.constant 6 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c6 step %c1 {
        %a0, %a1 = aie.objectfifo.acquire @a(Produce, 2) : memref<16xi32>, memref<16xi32>
        aie.objectfifo.release @a(Produce, 1)
      }
      aie.end
    }
    %core1 = aie.core(%other) {
      %c0 = arith.constant 0 : index
      %c6 = arith.constant 6 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c6 step %c1 {
        %b0, %b1 = aie.objectfifo.acquire @b(Produce, 2) : memref<16xi32>, memref<16xi32>
        aie.objectfifo.release @b(Produce, 1)
      }
      aie.end
    }
    %recv_a = aie.core(%dst_a) {
      %c0 = arith.constant 0 : index
      %c6 = arith.constant 6 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c6 step %c1 {
        %a0 = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @a(Consume, 1)
      }
      aie.end
    }
    %recv_b = aie.core(%dst_b) {
      %c0 = arith.constant 0 : index
      %c6 = arith.constant 6 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c6 step %c1 {
        %b0 = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @b(Consume, 1)
      }
      aie.end
    }
  }
}

// -----

// The assertion covers both objects this release sends inside a loop. Both
// the other FIFO's release and a later release of the same FIFO still poll.
// CHECK-LABEL: @assume_space
// CHECK: aie.core
// CHECK: scf.for
// CHECK-NOT: aie.dma_poll_for_queue_space
// CHECK: aie.dma_push_queue(MM2S : 0) bd_id 0
// CHECK-NOT: aie.dma_poll_for_queue_space
// CHECK: scf.index_switch
// CHECK-NOT: aie.dma_poll_for_queue_space
// CHECK: aie.dma_push_queue(MM2S : 0) bd_id 0
// CHECK-NOT: aie.dma_poll_for_queue_space
// CHECK: scf.index_switch
// CHECK: aie.dma_poll_for_queue_space(MM2S : 0)
// CHECK-NEXT: aie.dma_push_queue(MM2S : 0) bd_id 3
// CHECK: scf.index_switch
// CHECK: aie.dma_poll_for_queue_space(MM2S : 0)
// CHECK-NEXT: aie.dma_push_queue(MM2S : 0) bd_id 0
module @assume_space {
  aie.device(npu2) {
    %src = aie.tile(0, 2)
    %dst_a = aie.tile(0, 4)
    %dst_b = aie.tile(1, 4)
    aie.objectfifo @a(%src, {%dst_a}, [3, 2]) {packet, packet_id = 3 : i8, prod_dma_channel = 0 : i32, release_strategy = #aie.release_strategy<push_queue>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%src, {%dst_b}, [2, 2]) {packet, packet_id = 4 : i8, prod_dma_channel = 0 : i32, release_strategy = #aie.release_strategy<push_queue>} : !aie.objectfifo<memref<16xi32>>
    %core0 = aie.core(%src) {
      %c0 = arith.constant 0 : index
      %c6 = arith.constant 6 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c6 step %c1 {
        %a0, %a1 = aie.objectfifo.acquire @a(Produce, 2) : memref<16xi32>, memref<16xi32>
        aie.objectfifo.release {assume_queue_space} @a(Produce, 2)
        %b0 = aie.objectfifo.acquire @b(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @b(Produce, 1)
        %next_a, %next_a1 = aie.objectfifo.acquire @a(Produce, 2) : memref<16xi32>, memref<16xi32>
        aie.objectfifo.release @a(Produce, 1)
      }
      aie.end
    }
    %recv_a = aie.core(%dst_a) {
      %c0 = arith.constant 0 : index
      %c6 = arith.constant 6 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c6 step %c1 {
        %a0 = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @a(Consume, 1)
      }
      aie.end
    }
    %recv_b = aie.core(%dst_b) {
      %c0 = arith.constant 0 : index
      %c6 = arith.constant 6 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c6 step %c1 {
        %b0 = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @b(Consume, 1)
      }
      aie.end
    }
  }
}

