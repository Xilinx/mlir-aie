//===- 06-aie1.mlir --------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DESIGN SKETCH -- not a lit test. See 00-model.mlir first.
//
// Binary-lock devices. Checked against
// test/objectFifo-stateful-transform/base/base_test_AIE1.mlir.
//
//===----------------------------------------------------------------------===//
//
// LOCKS ROTATE WITH THE BUFFERS
//
// A binary-lock pool carries one lock per BUFFER and has a single implicit
// segment spanning the object:
//
//   locks = [@of1_lock_0, @of1_lock_1]      locks[i] guards buffers[i]
//
// Both actors take the same lock for a given buffer. The lock value identifies
// ownership: 1 for a fill-release or a drain-acquire, 0 otherwise.
//
// A pool carries either `locks` or `segments`, matching the device's lock kind.
// The two are indexed on different axes -- `locks` follows the buffer rotation,
// `segments` partitions an object -- so a pool never carries both.
//
// Join and distribute need one lock pair per segment, so they require semaphore
// locks and do not appear on binary-lock devices.
//
//===----------------------------------------------------------------------===//
//
// BD EMISSION
//
//     for each buffer index i:
//       acquire(locks[i], drains ? 1 : 0)
//       dma_bd(buffers[i])
//       release(locks[i], drains ? 0 : 1)
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// STAGE 2 -- after --aie-objectfifo-allocate
//===----------------------------------------------------------------------===//

module @stage2 {
  aie.device(xcvc1902) {
    %tile12 = aie.tile(1, 2)
    %tile33 = aie.tile(3, 3)

    %b0 = aie.buffer(%tile12) {sym_name = "of1_buff_0"} : memref<16xi32>
    %b1 = aie.buffer(%tile12) {sym_name = "of1_buff_1"} : memref<16xi32>
    %l0 = aie.lock(%tile12) {init = 0 : i32, sym_name = "of1_lock_0"}
    %l1 = aie.lock(%tile12) {init = 0 : i32, sym_name = "of1_lock_1"}

    %cb0 = aie.buffer(%tile33) {sym_name = "of1_cons_buff_0"} : memref<16xi32>
    %cb1 = aie.buffer(%tile33) {sym_name = "of1_cons_buff_1"} : memref<16xi32>
    %cl0 = aie.lock(%tile33) {init = 0 : i32, sym_name = "of1_cons_lock_0"}
    %cl1 = aie.lock(%tile33) {init = 0 : i32, sym_name = "of1_cons_lock_1"}

    aie.objectfifo.pool @of1_prod_pool(%tile12) {
      depth   = 2 : i32,
      buffers = [@of1_buff_0, @of1_buff_1],
      locks   = [@of1_lock_0, @of1_lock_1]
    } : memref<16xi32>

    aie.objectfifo.pool @of1_cons_pool(%tile33) {
      depth   = 2 : i32,
      buffers = [@of1_cons_buff_0, @of1_cons_buff_1],
      locks   = [@of1_cons_lock_0, @of1_cons_lock_1]
    } : memref<16xi32>

    aie.objectfifo.core_endpoint @of1_prod_core(%tile12) fills  @of1_prod_pool
    aie.objectfifo.dma_endpoint  @of1_prod_dma(%tile12)  drains @of1_prod_pool {channelIndex = 0}
    aie.objectfifo.dma_endpoint  @of1_cons_dma(%tile33)  fills  @of1_cons_pool  {channelIndex = 0}
    aie.objectfifo.core_endpoint @of1_cons_core(%tile33) drains @of1_cons_pool

    aie.flow(%tile12, DMA : 0, %tile33, DMA : 0)
  }
}

//===----------------------------------------------------------------------===//
// STAGE 3 -- after --aie-objectfifo-lower-dmas
//
// Each BD acquires and releases the lock belonging to the buffer it addresses.
//===----------------------------------------------------------------------===//

module @stage3 {
  aie.device(xcvc1902) {
    %mem12 = aie.mem(%tile12) {
      aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:
      aie.use_lock(%l0, Acquire, 1)
      aie.dma_bd(%b0 : memref<16xi32>, 0, 16)
      aie.use_lock(%l0, Release, 0)
      aie.next_bd ^bb2
    ^bb2:
      aie.use_lock(%l1, Acquire, 1)
      aie.dma_bd(%b1 : memref<16xi32>, 0, 16)
      aie.use_lock(%l1, Release, 0)
      aie.next_bd ^bb1
    ^bb3:
      aie.end
    }
  }
}
