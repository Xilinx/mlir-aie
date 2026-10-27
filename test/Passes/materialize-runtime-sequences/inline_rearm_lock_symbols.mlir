// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-materialize-runtime-sequences %s | FileCheck %s
// RUN: aie-opt --aie-materialize-runtime-sequences --aie-lower-dma-channel-reset %s | FileCheck %s --check-prefix=LOWER

// The runtime sequence names only the binding. Its lock-array symbols must
// recursively pull in both locks and rewrite a colliding lock to the clone,
// rather than accidentally binding to the caller's unrelated @producer.
module {
  // CHECK-LABEL: aie.device(npu2) @caller
  // CHECK: %[[TILE:.*]] = aie.tile(0, 3)
  // CHECK-DAG: aie.lock @producer(%[[TILE]], 8) {init = 7 : i32}
  // CHECK-DAG: aie.lock @producer_0(%[[TILE]], 0) {init = 2 : i32}
  // CHECK-DAG: aie.lock @consumer(%[[TILE]], 1) {init = 0 : i32}
  // CHECK-DAG: aie.objectfifo_rearm_binding @rearm_binding channels(%[[TILE]] : index) locks([@producer_0, @consumer]) {{.*}}lock_inits = array<i32: 2, 0>
  // CHECK: aie.runtime_sequence @entry(
  // CHECK: aiex.npu.load_pdi {device_ref = @configuration}
  // CHECK: aiex.dma_channel_reset_for(@rearm_binding)

  // LOWER-LABEL: aie.device(npu2) @caller
  // LOWER-DAG: aie.lock @producer({{.*}}, 8) {init = 7 : i32}
  // LOWER-DAG: aie.lock @producer_0({{.*}}, 0) {init = 2 : i32}
  // LOWER-DAG: aie.lock @consumer({{.*}}, 1) {init = 0 : i32}
  // LOWER: aie.runtime_sequence @entry(
  // LOWER: aiex.npu.load_pdi {device_ref = @configuration}
  // LOWER: aiex.set_lock(@producer_0, %{{.*}})
  // LOWER: aiex.set_lock(@consumer, %{{.*}})
  aie.device(npu2) @caller {
    %tile = aie.tile(0, 3)
    aie.lock @producer(%tile, 8) {init = 7 : i32}
    aie.runtime_sequence @entry(%arg0: memref<1xi32>) {
      aiex.configure @configuration {
        aiex.run @rearm(%arg0) : (memref<1xi32>)
      }
    }
  }

  // The source declaration and its references remain unchanged.
  // CHECK-LABEL: aie.device(npu2) @configuration
  // CHECK: aie.lock @producer({{.*}}, 0) {init = 2 : i32}
  // CHECK: aie.lock @consumer({{.*}}, 1) {init = 0 : i32}
  // CHECK: aie.objectfifo_rearm_binding @rearm_binding channels({{.*}}) locks([@producer, @consumer])
  // LOWER-LABEL: aie.device(npu2) @configuration
  aie.device(npu2) @configuration {
    %tile = aie.tile(0, 3)
    aie.lock @producer(%tile, 0) {init = 2 : i32}
    aie.lock @consumer(%tile, 1) {init = 0 : i32}
    aie.objectfifo_rearm_binding @rearm_binding channels(%tile : index) locks([@producer, @consumer]) {channel_dirs = array<i32: 0>, channel_indices = array<i32: 0>, head_bd_ids = array<i32: 5>, lock_inits = array<i32: 2, 0>, repeat_counts = array<i32: 0>}
    aie.runtime_sequence @rearm(%arg0: memref<1xi32>) {
      aiex.dma_channel_reset_for(@rearm_binding)
    }
  }
}
