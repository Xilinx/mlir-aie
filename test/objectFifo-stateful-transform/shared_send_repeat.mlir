// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --split-input-file --aie-objectFifo-stateful-transform="skip-verify=true" --aie-assign-bd-ids %s | FileCheck %s

// CHECK-LABEL: @repeat_depth1
// CHECK: aie.dma_push_queue(MM2S : 0) bd_id 0 {{.*}}repeat_count = 2 : i32}
// CHECK: aie.dma_start(MM2S, 0, {{.*}}) {core_pushes}
// CHECK: aie.dma_bd(%a__buf0 {{.*}}bd_id = 0 : i32
// CHECK-NOT: aie.dma_bd(%a__buf
// CHECK: aie.dma_start(S2MM
module @repeat_depth1 {
  aie.device(npu2) {
    %src = aie.tile(0, 2)
    %dst = aie.tile(0, 4)
    aie.objectfifo @a(%src, {%dst}, [1, 2]) {repeat_count = 3 : i32, release_strategy = #aie.release_strategy<push_queue>} : !aie.objectfifo<memref<16xi32>>
    %producer = aie.core(%src) {
      %a0 = aie.objectfifo.acquire @a(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a(Produce, 1)
      %a1 = aie.objectfifo.acquire @a(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a(Produce, 1)
      aie.end
    }
  }
}

// -----

// CHECK-LABEL: @repeat_depth2
// CHECK: aie.dma_push_queue(MM2S : 0) bd_id 0 {{.*}}repeat_count = 2 : i32}
// CHECK: aie.dma_push_queue(MM2S : 0) bd_id 1 {{.*}}repeat_count = 2 : i32}
// CHECK: aie.dma_start(MM2S, 0, {{.*}}) {core_pushes}
// CHECK: aie.dma_bd(%a__buf0 {{.*}}bd_id = 0 : i32
// CHECK: aie.dma_bd(%a__buf1 {{.*}}bd_id = 1 : i32
// CHECK-NOT: aie.dma_bd(%a__buf
// CHECK: aie.dma_start(S2MM
module @repeat_depth2 {
  aie.device(npu2) {
    %src = aie.tile(0, 2)
    %dst = aie.tile(0, 4)
    aie.objectfifo @a(%src, {%dst}, [2, 2]) {repeat_count = 3 : i32, release_strategy = #aie.release_strategy<push_queue>} : !aie.objectfifo<memref<16xi32>>
    %producer = aie.core(%src) {
      %a0 = aie.objectfifo.acquire @a(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a(Produce, 1)
      %a1 = aie.objectfifo.acquire @a(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a(Produce, 1)
      aie.end
    }
  }
}

// -----

// CHECK-LABEL: @repeat_depth2_bounded
// CHECK: aie.dma_push_queue(MM2S : 0) bd_id 0 {{.*}}repeat_count = 2 : i32}
// CHECK: aie.dma_push_queue(MM2S : 0) bd_id 1 {{.*}}repeat_count = 2 : i32}
// CHECK: aie.dma_start(MM2S, 0, {{.*}}) {core_pushes}
// CHECK: aie.dma_bd(%a__buf0 {{.*}}bd_id = 0 : i32
// CHECK: aie.dma_bd(%a__buf1 {{.*}}bd_id = 1 : i32
// CHECK-NOT: aie.dma_bd(%a__buf
// CHECK: aie.dma_start(S2MM
module @repeat_depth2_bounded {
  aie.device(npu2) {
    %src = aie.tile(0, 2)
    %dst = aie.tile(0, 4)
    aie.objectfifo @a(%src, {%dst}, [2, 2]) {repeat_count = 3 : i32, release_strategy = #aie.release_strategy<push_queue>, iter_count = 2 : i32} : !aie.objectfifo<memref<16xi32>>
    %producer = aie.core(%src) {
      %a0 = aie.objectfifo.acquire @a(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a(Produce, 1)
      %a1 = aie.objectfifo.acquire @a(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a(Produce, 1)
      aie.end
    }
  }
}
