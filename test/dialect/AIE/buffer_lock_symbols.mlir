// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s | FileCheck %s
// RUN: aie-opt --symbol-dce %s | FileCheck %s --check-prefix=DCE

module {
  aie.device(npu1) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)
    // CHECK: aie.buffer @live(
    // CHECK-SAME: {address = 1024 : i32, sym_visibility = "private"} : memref<16xi32>
    aie.buffer @live(%tile) {address = 1024 : i32, sym_visibility = "private"} : memref<16xi32>
    aie.buffer @dead(%tile) {sym_visibility = "private"} : memref<16xi32>
    // CHECK: aie.external_buffer @external_live {sym_visibility = "private"} : memref<16xi32>
    aie.external_buffer @external_live {sym_visibility = "private"} : memref<16xi32>
    aie.external_buffer @external_dead {sym_visibility = "private"} : memref<16xi32>
    aie.external_buffer @registered_live {sym_visibility = "private"} : memref<16xi32>
    aie.external_buffer @registered_dead {sym_visibility = "private"} : memref<16xi32>
    // CHECK: aie.lock @live_lock(
    // CHECK-SAME: , 0) {init = 1 : i32, sym_visibility = "private"}
    aie.lock @live_lock(%tile, 0) {init = 1 : i32, sym_visibility = "private"}
    aie.lock @dead_lock(%tile, 1) {sym_visibility = "private"}
    aie.lock @runtime_lock(%tile, 2) {init = 0 : i32, sym_visibility = "private"}

    aie.objectfifo @registered(%shim, {%tile}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    // Registration is metadata and must not require a device-level SSA accessor.
    // CHECK: aie.objectfifo.register_external_buffers @registered({{.*}}, [@registered_live])
    aie.objectfifo.register_external_buffers @registered (%shim, [@registered_live])

    // CHECK: aie.core(
    aie.core(%tile) {
      // CHECK: %[[CORE_BUFFER:.*]] = aie.get_buffer @live : memref<16xi32>
      %buffer = aie.get_buffer @live : memref<16xi32>
      %value = arith.constant 1 : i32
      %index = arith.constant 0 : index
      // CHECK: aie.use_lock(@live_lock, AcquireGreaterEqual, %{{.*}})
      aie.use_lock(@live_lock, AcquireGreaterEqual, %value)
      // CHECK: %[[LOADED:.*]] = memref.load %[[CORE_BUFFER]][
      %loaded = memref.load %buffer[%index] : memref<16xi32>
      // CHECK: memref.store %[[LOADED]], %[[CORE_BUFFER]][
      memref.store %loaded, %buffer[%index] : memref<16xi32>
      aie.use_lock(@live_lock, Release, %value)
      aie.end
    }

    // The DMA region obtains its own SSA value, not one captured from the core.
    // CHECK: aie.mem(
    aie.mem(%tile) {
      aie.dma_start(MM2S, 0, ^bd, ^end)
    ^bd:
      // CHECK: %[[DMA_BUFFER:.*]] = aie.get_buffer @live : memref<16xi32>
      %buffer = aie.get_buffer @live : memref<16xi32>
      // CHECK: aie.dma_bd(%[[DMA_BUFFER]] : memref<16xi32>
      aie.dma_bd(%buffer : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^bd
    ^end:
      aie.end
    }

    // CHECK: aie.shim_dma(
    aie.shim_dma(%shim) {
      aie.dma_start(S2MM, 0, ^bd, ^end)
    ^bd:
      // CHECK: %[[EXTERNAL:.*]] = aie.get_buffer @external_live : memref<16xi32>
      %external = aie.get_buffer @external_live : memref<16xi32>
      // CHECK: aie.dma_bd(%[[EXTERNAL]] : memref<16xi32>
      aie.dma_bd(%external : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^bd
    ^end:
      aie.end
    }

    // CHECK: aie.runtime_sequence @set_runtime_lock()
    aie.runtime_sequence @set_runtime_lock() {
      %value = arith.constant 1 : i32
      // CHECK: aiex.set_lock(@runtime_lock, %{{.*}})
      aiex.set_lock(@runtime_lock, %value)
    }
  }
}

// DCE: aie.buffer @live(
// DCE-NOT: aie.buffer @dead(
// DCE: aie.external_buffer @external_live
// DCE-NOT: aie.external_buffer @external_dead
// DCE: aie.external_buffer @registered_live
// DCE-NOT: aie.external_buffer @registered_dead
// DCE: aie.lock @live_lock(
// DCE-NOT: aie.lock @dead_lock(
// DCE: aie.lock @runtime_lock(
// DCE: aie.objectfifo.register_external_buffers @registered({{.*}}, [@registered_live])
// DCE-NOT: aie.get_buffer
// DCE: aie.core(
// DCE: aie.get_buffer @live : memref<16xi32>
// DCE: aie.use_lock(@live_lock, AcquireGreaterEqual,
// DCE: aie.mem(
// DCE: aie.get_buffer @live : memref<16xi32>
// DCE: aie.shim_dma(
// DCE: aie.get_buffer @external_live : memref<16xi32>
// DCE: aie.runtime_sequence @set_runtime_lock()
// DCE: aiex.set_lock(@runtime_lock, %{{.*}})
