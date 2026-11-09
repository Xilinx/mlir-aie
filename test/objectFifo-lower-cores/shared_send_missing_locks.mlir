// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --split-input-file --aie-objectfifo-lower-cores --canonicalize %s | FileCheck %s

// CHECK-LABEL: module @no_locks
// CHECK: aie.core
// CHECK: aie.dma_poll_for_queue_space(MM2S : 0)
// CHECK-NEXT: aie.dma_push_queue(MM2S : 0) {{.*}}bd_tag = 0 : i32
// CHECK: aie.dma_poll_for_queue_space(MM2S : 0)
// CHECK-NEXT: aie.dma_push_queue(MM2S : 0) {{.*}}bd_tag = 1 : i32
// CHECK-NOT: aie.dma_poll_for_queue_space
// CHECK: aie.dma_push_queue(MM2S : 1) {{.*}}bd_tag = 2 : i32
// CHECK-NOT: aie.dma_poll_for_queue_space
// CHECK: aie.dma_push_queue(MM2S : 0) {{.*}}bd_tag = 0 : i32
// CHECK-NOT: aie.dma_poll_for_queue_space
// CHECK: aie.end
module @no_locks {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %free = aie.lock(%tile) {sym_name = "free", init = 1 : i32}
    %full = aie.lock(%tile) {sym_name = "full", init = 0 : i32}
    %b0 = aie.buffer(%tile) {sym_name = "b0"} : memref<16xi32>
    aie.objectfifo.pool @p0(%tile) {depth = 1 : i32, buffers = [@b0]} : memref<16xi32> {
      aie.objectfifo.segment @all {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.core_endpoint @f0(%tile) fills @p0 {core_push = #aie.core_push<channel = 0, bd_tags = [0]>}
    %b1 = aie.buffer(%tile) {sym_name = "b1"} : memref<16xi32>
    aie.objectfifo.pool @p1(%tile) {depth = 1 : i32, buffers = [@b1]} : memref<16xi32> {
      aie.objectfifo.segment @all {offset = 0 : i32, size = 16 : i32, produceLock = @free, consumeLock = @full}
    }
    aie.objectfifo.core_endpoint @f1(%tile) fills @p1 {core_push = #aie.core_push<channel = 0, bd_tags = [1]>}
    %b2 = aie.buffer(%tile) {sym_name = "b2"} : memref<16xi32>
    aie.objectfifo.pool @p2(%tile) {depth = 1 : i32, buffers = [@b2]} : memref<16xi32> {
      aie.objectfifo.segment @all {offset = 0 : i32, size = 16 : i32, produceLock = @free, consumeLock = @full}
    }
    aie.objectfifo.core_endpoint @f2(%tile) fills @p2 {core_push = #aie.core_push<channel = 1, bd_tags = [2]>}
    %core = aie.core(%tile) {
      %obj0 = aie.objectfifo.acquire @f0(1) : memref<16xi32>
      aie.objectfifo.release @f0(1)
      %obj1 = aie.objectfifo.acquire @f1(1) : memref<16xi32>
      aie.objectfifo.release @f1(1)
      %obj2 = aie.objectfifo.acquire @f2(1) : memref<16xi32>
      aie.objectfifo.release @f2(1)
      %obj3 = aie.objectfifo.acquire @f0(1) : memref<16xi32>
      aie.objectfifo.release {assume_queue_space} @f0(1)
      aie.end
    }
  }
}

// -----

// CHECK-LABEL: module @no_produce_lock
// CHECK: aie.core
// CHECK: aie.dma_poll_for_queue_space(MM2S : 0)
// CHECK-NEXT: aie.dma_push_queue(MM2S : 0) {{.*}}bd_tag = 0 : i32
// CHECK: aie.dma_poll_for_queue_space(MM2S : 0)
// CHECK-NEXT: aie.dma_push_queue(MM2S : 0) {{.*}}bd_tag = 1 : i32
// CHECK-NOT: aie.dma_poll_for_queue_space
// CHECK: aie.dma_push_queue(MM2S : 1) {{.*}}bd_tag = 2 : i32
// CHECK-NOT: aie.dma_poll_for_queue_space
// CHECK: aie.dma_push_queue(MM2S : 0) {{.*}}bd_tag = 0 : i32
// CHECK-NOT: aie.dma_poll_for_queue_space
// CHECK: aie.end
module @no_produce_lock {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %free = aie.lock(%tile) {sym_name = "free", init = 1 : i32}
    %full = aie.lock(%tile) {sym_name = "full", init = 0 : i32}
    %b0 = aie.buffer(%tile) {sym_name = "b0"} : memref<16xi32>
    aie.objectfifo.pool @p0(%tile) {depth = 1 : i32, buffers = [@b0]} : memref<16xi32> {
      aie.objectfifo.segment @all {offset = 0 : i32, size = 16 : i32, consumeLock = @full}
    }
    aie.objectfifo.core_endpoint @f0(%tile) fills @p0 {core_push = #aie.core_push<channel = 0, bd_tags = [0]>}
    %b1 = aie.buffer(%tile) {sym_name = "b1"} : memref<16xi32>
    aie.objectfifo.pool @p1(%tile) {depth = 1 : i32, buffers = [@b1]} : memref<16xi32> {
      aie.objectfifo.segment @all {offset = 0 : i32, size = 16 : i32, produceLock = @free, consumeLock = @full}
    }
    aie.objectfifo.core_endpoint @f1(%tile) fills @p1 {core_push = #aie.core_push<channel = 0, bd_tags = [1]>}
    %b2 = aie.buffer(%tile) {sym_name = "b2"} : memref<16xi32>
    aie.objectfifo.pool @p2(%tile) {depth = 1 : i32, buffers = [@b2]} : memref<16xi32> {
      aie.objectfifo.segment @all {offset = 0 : i32, size = 16 : i32, produceLock = @free, consumeLock = @full}
    }
    aie.objectfifo.core_endpoint @f2(%tile) fills @p2 {core_push = #aie.core_push<channel = 1, bd_tags = [2]>}
    %core = aie.core(%tile) {
      %obj0 = aie.objectfifo.acquire @f0(1) : memref<16xi32>
      aie.objectfifo.release @f0(1)
      %obj1 = aie.objectfifo.acquire @f1(1) : memref<16xi32>
      aie.objectfifo.release @f1(1)
      %obj2 = aie.objectfifo.acquire @f2(1) : memref<16xi32>
      aie.objectfifo.release @f2(1)
      %obj3 = aie.objectfifo.acquire @f0(1) : memref<16xi32>
      aie.objectfifo.release {assume_queue_space} @f0(1)
      aie.end
    }
  }
}

// -----

// CHECK-LABEL: module @no_consume_lock
// CHECK: aie.core
// CHECK: aie.dma_poll_for_queue_space(MM2S : 0)
// CHECK-NEXT: aie.dma_push_queue(MM2S : 0) {{.*}}bd_tag = 0 : i32
// CHECK: aie.dma_poll_for_queue_space(MM2S : 0)
// CHECK-NEXT: aie.dma_push_queue(MM2S : 0) {{.*}}bd_tag = 1 : i32
// CHECK-NOT: aie.dma_poll_for_queue_space
// CHECK: aie.dma_push_queue(MM2S : 1) {{.*}}bd_tag = 2 : i32
// CHECK-NOT: aie.dma_poll_for_queue_space
// CHECK: aie.dma_push_queue(MM2S : 0) {{.*}}bd_tag = 0 : i32
// CHECK-NOT: aie.dma_poll_for_queue_space
// CHECK: aie.end
module @no_consume_lock {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %free = aie.lock(%tile) {sym_name = "free", init = 1 : i32}
    %full = aie.lock(%tile) {sym_name = "full", init = 0 : i32}
    %b0 = aie.buffer(%tile) {sym_name = "b0"} : memref<16xi32>
    aie.objectfifo.pool @p0(%tile) {depth = 1 : i32, buffers = [@b0]} : memref<16xi32> {
      aie.objectfifo.segment @all {offset = 0 : i32, size = 16 : i32, produceLock = @free}
    }
    aie.objectfifo.core_endpoint @f0(%tile) fills @p0 {core_push = #aie.core_push<channel = 0, bd_tags = [0]>}
    %b1 = aie.buffer(%tile) {sym_name = "b1"} : memref<16xi32>
    aie.objectfifo.pool @p1(%tile) {depth = 1 : i32, buffers = [@b1]} : memref<16xi32> {
      aie.objectfifo.segment @all {offset = 0 : i32, size = 16 : i32, produceLock = @free, consumeLock = @full}
    }
    aie.objectfifo.core_endpoint @f1(%tile) fills @p1 {core_push = #aie.core_push<channel = 0, bd_tags = [1]>}
    %b2 = aie.buffer(%tile) {sym_name = "b2"} : memref<16xi32>
    aie.objectfifo.pool @p2(%tile) {depth = 1 : i32, buffers = [@b2]} : memref<16xi32> {
      aie.objectfifo.segment @all {offset = 0 : i32, size = 16 : i32, produceLock = @free, consumeLock = @full}
    }
    aie.objectfifo.core_endpoint @f2(%tile) fills @p2 {core_push = #aie.core_push<channel = 1, bd_tags = [2]>}
    %core = aie.core(%tile) {
      %obj0 = aie.objectfifo.acquire @f0(1) : memref<16xi32>
      aie.objectfifo.release @f0(1)
      %obj1 = aie.objectfifo.acquire @f1(1) : memref<16xi32>
      aie.objectfifo.release @f1(1)
      %obj2 = aie.objectfifo.acquire @f2(1) : memref<16xi32>
      aie.objectfifo.release @f2(1)
      %obj3 = aie.objectfifo.acquire @f0(1) : memref<16xi32>
      aie.objectfifo.release {assume_queue_space} @f0(1)
      aie.end
    }
  }
}

// -----

// CHECK-LABEL: module @all_locks
// CHECK: aie.core
// CHECK-NOT: aie.dma_poll_for_queue_space
// CHECK: aie.dma_push_queue(MM2S : 0) {{.*}}bd_tag = 0 : i32
// CHECK-NOT: aie.dma_poll_for_queue_space
// CHECK: aie.dma_push_queue(MM2S : 0) {{.*}}bd_tag = 1 : i32
// CHECK-NOT: aie.dma_poll_for_queue_space
// CHECK: aie.dma_push_queue(MM2S : 1) {{.*}}bd_tag = 2 : i32
// CHECK-NOT: aie.dma_poll_for_queue_space
// CHECK: aie.dma_push_queue(MM2S : 0) {{.*}}bd_tag = 0 : i32
// CHECK-NOT: aie.dma_poll_for_queue_space
// CHECK: aie.end
module @all_locks {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %free = aie.lock(%tile) {sym_name = "free", init = 1 : i32}
    %full = aie.lock(%tile) {sym_name = "full", init = 0 : i32}
    %b0 = aie.buffer(%tile) {sym_name = "b0"} : memref<16xi32>
    aie.objectfifo.pool @p0(%tile) {depth = 1 : i32, buffers = [@b0]} : memref<16xi32> {
      aie.objectfifo.segment @all {offset = 0 : i32, size = 16 : i32, produceLock = @free, consumeLock = @full}
    }
    aie.objectfifo.core_endpoint @f0(%tile) fills @p0 {core_push = #aie.core_push<channel = 0, bd_tags = [0]>}
    %b1 = aie.buffer(%tile) {sym_name = "b1"} : memref<16xi32>
    aie.objectfifo.pool @p1(%tile) {depth = 1 : i32, buffers = [@b1]} : memref<16xi32> {
      aie.objectfifo.segment @all {offset = 0 : i32, size = 16 : i32, produceLock = @free, consumeLock = @full}
    }
    aie.objectfifo.core_endpoint @f1(%tile) fills @p1 {core_push = #aie.core_push<channel = 0, bd_tags = [1]>}
    %b2 = aie.buffer(%tile) {sym_name = "b2"} : memref<16xi32>
    aie.objectfifo.pool @p2(%tile) {depth = 1 : i32, buffers = [@b2]} : memref<16xi32> {
      aie.objectfifo.segment @all {offset = 0 : i32, size = 16 : i32, produceLock = @free, consumeLock = @full}
    }
    aie.objectfifo.core_endpoint @f2(%tile) fills @p2 {core_push = #aie.core_push<channel = 1, bd_tags = [2]>}
    %core = aie.core(%tile) {
      %obj0 = aie.objectfifo.acquire @f0(1) : memref<16xi32>
      aie.objectfifo.release @f0(1)
      %obj1 = aie.objectfifo.acquire @f1(1) : memref<16xi32>
      aie.objectfifo.release @f1(1)
      %obj2 = aie.objectfifo.acquire @f2(1) : memref<16xi32>
      aie.objectfifo.release @f2(1)
      %obj3 = aie.objectfifo.acquire @f0(1) : memref<16xi32>
      aie.objectfifo.release {assume_queue_space} @f0(1)
      aie.end
    }
  }
}
