//===- hoist-bfp16-conversions.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file --optimize-aievec="aie-target=aie2p" | FileCheck %s

// The loops multiply the 4 tiles of q by blocks of k loaded as bfp16, and do
// not modify q. q is converted once before the loops into a copy of bfp16
// blocks, written through a store stream, and the loops load from the copy.
// The loads of k become a load stream.
// CHECK-LABEL: aie.device(npu2) @hoisted
// CHECK: %[[QB:.*]] = aie.buffer({{.*}}) {sym_name = "q_bfp16"} : memref<288xi8>
// CHECK: aie.core
// CHECK: %[[A0:.*]], %[[S0:.*]], %[[C0:.*]] = aievec.fifo_start %[[QB]][%{{.*}}]
// CHECK-COUNT-4: vector.transfer_read {{.*}} : memref<256xbf16>, vector<64xbf16>
// CHECK: aievec.to_bfp16
// CHECK: aievec.fifo_st_push_bfp16 %[[A0]], %[[S0]], %[[C0]]
// CHECK-COUNT-3: aievec.fifo_st_push_bfp16
// CHECK: scf.for
// CHECK: aievec.fifo_start %kb
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (!llvm.ptr, vector<32xi32>, i32)
// CHECK: aievec.load_bfp16 %[[QB]]
// CHECK: aievec.fifo_ld_pop_bfp16
// CHECK: aievec.matmul_aie2p_bfp16
// CHECK-NOT: aievec.to_bfp16
aie.device(npu2) @hoisted {
  %t = aie.tile(0, 2)
  %q = aie.buffer(%t) {sym_name = "q"} : memref<4x64xbf16>
  %kb = aie.buffer(%t) {sym_name = "kb"} : memref<288xi8>
  %acc = aie.buffer(%t) {sym_name = "acc"} : memref<4x4x64xf32>
  %lk = aie.lock(%t, 0) {init = 0 : i32}
  %lkp = aie.lock(%t, 1) {init = 1 : i32}
  %m = aie.mem(%t) {
    %0 = aie.dma_start(S2MM, 0, ^bd, ^end)
  ^bd:
    %m1 = arith.constant 1 : i32
    aie.use_lock(%lkp, AcquireGreaterEqual, %m1)
    aie.dma_bd(%kb : memref<288xi8> offset = 0 len = 288)
    aie.use_lock(%lk, Release, %m1)
    aie.next_bd ^bd
  ^end:
    aie.end
  }
  %c = aie.core(%t) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c72 = arith.constant 72 : index
    %bz = arith.constant 0.0 : bf16
    %fz = arith.constant 0.0 : f32
    %one = arith.constant 1 : i32
    scf.for %blk = %c0 to %c16 step %c1 {
      aie.use_lock(%lk, AcquireGreaterEqual, %one)
      scf.for %i = %c0 to %c4 step %c1 {
        scf.for %j = %c0 to %c4 step %c1 {
          %qv = vector.transfer_read %q[%i, %c0], %bz {in_bounds = [true]} : memref<4x64xbf16>, vector<64xbf16>
          %qt = vector.shape_cast %qv : vector<64xbf16> to vector<8x8xbf16>
          %qm, %qe = aievec.to_bfp16 %qt : vector<8x8xbf16> -> vector<64xi8>, vector<8xi8>
          %ko = arith.muli %j, %c72 : index
          %km, %ke = aievec.load_bfp16 %kb[%ko] : memref<288xi8> -> vector<64xi8>, vector<8xi8>
          %av = vector.transfer_read %acc[%i, %j, %c0], %fz {in_bounds = [true]} : memref<4x4x64xf32>, vector<64xf32>
          %a = vector.shape_cast %av : vector<64xf32> to vector<8x8xf32>
          %r = aievec.matmul_aie2p_bfp16 %qm, %qe, %km, %ke, %a : vector<64xi8>, vector<8xi8>, vector<64xi8>, vector<8xi8> into vector<8x8xf32>
          %rv = vector.shape_cast %r : vector<8x8xf32> to vector<64xf32>
          vector.transfer_write %rv, %acc[%i, %j, %c0] {in_bounds = [true]} : vector<64xf32>, memref<4x4x64xf32>
        }
      }
      aie.use_lock(%lkp, Release, %one)
    }
    aie.end
  }
}

// -----

// k is also converted in the loop, so q is not hoisted.
// CHECK-LABEL: aie.device(npu2) @not_partnered
// CHECK-NOT: q_bfp16
aie.device(npu2) @not_partnered {
  %t = aie.tile(0, 2)
  %q = aie.buffer(%t) {sym_name = "q"} : memref<4x64xbf16>
  %kb = aie.buffer(%t) {sym_name = "kb"} : memref<288xi8>
  %k = aie.buffer(%t) {sym_name = "k"} : memref<4x64xbf16>
  %acc = aie.buffer(%t) {sym_name = "acc"} : memref<4x4x64xf32>
  %lk = aie.lock(%t, 0) {init = 0 : i32}
  %lkp = aie.lock(%t, 1) {init = 1 : i32}
  %m = aie.mem(%t) {
    %0 = aie.dma_start(S2MM, 0, ^bd, ^end)
  ^bd:
    %m1 = arith.constant 1 : i32
    aie.use_lock(%lkp, AcquireGreaterEqual, %m1)
    aie.dma_bd(%kb : memref<288xi8> offset = 0 len = 288)
    aie.use_lock(%lk, Release, %m1)
    aie.next_bd ^bd
  ^end:
    aie.end
  }
  %c = aie.core(%t) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c72 = arith.constant 72 : index
    %bz = arith.constant 0.0 : bf16
    %fz = arith.constant 0.0 : f32
    %one = arith.constant 1 : i32
    scf.for %blk = %c0 to %c16 step %c1 {
      aie.use_lock(%lk, AcquireGreaterEqual, %one)
      scf.for %i = %c0 to %c4 step %c1 {
        scf.for %j = %c0 to %c4 step %c1 {
          %qv = vector.transfer_read %q[%i, %c0], %bz {in_bounds = [true]} : memref<4x64xbf16>, vector<64xbf16>
          %qt = vector.shape_cast %qv : vector<64xbf16> to vector<8x8xbf16>
          %qm, %qe = aievec.to_bfp16 %qt : vector<8x8xbf16> -> vector<64xi8>, vector<8xi8>
          %kv = vector.transfer_read %k[%j, %c0], %bz {in_bounds = [true]} : memref<4x64xbf16>, vector<64xbf16>
          %kt = vector.shape_cast %kv : vector<64xbf16> to vector<8x8xbf16>
          %km, %ke = aievec.to_bfp16 %kt transpose : vector<8x8xbf16> -> vector<64xi8>, vector<8xi8>
          %av = vector.transfer_read %acc[%i, %j, %c0], %fz {in_bounds = [true]} : memref<4x4x64xf32>, vector<64xf32>
          %a = vector.shape_cast %av : vector<64xf32> to vector<8x8xf32>
          %r = aievec.matmul_aie2p_bfp16 %qm, %qe, %km, %ke, %a : vector<64xi8>, vector<8xi8>, vector<64xi8>, vector<8xi8> into vector<8x8xf32>
          %rv = vector.shape_cast %r : vector<8x8xf32> to vector<64xf32>
          vector.transfer_write %rv, %acc[%i, %j, %c0] {in_bounds = [true]} : vector<64xf32>, memref<4x4x64xf32>
        }
      }
      aie.use_lock(%lkp, Release, %one)
    }
    aie.end
  }
}

// -----

// The loops write q, so q is not hoisted.
// CHECK-LABEL: aie.device(npu2) @written
// CHECK-NOT: q_bfp16
aie.device(npu2) @written {
  %t = aie.tile(0, 2)
  %q = aie.buffer(%t) {sym_name = "q"} : memref<4x64xbf16>
  %kb = aie.buffer(%t) {sym_name = "kb"} : memref<288xi8>
  %acc = aie.buffer(%t) {sym_name = "acc"} : memref<4x4x64xf32>
  %lk = aie.lock(%t, 0) {init = 0 : i32}
  %lkp = aie.lock(%t, 1) {init = 1 : i32}
  %m = aie.mem(%t) {
    %0 = aie.dma_start(S2MM, 0, ^bd, ^end)
  ^bd:
    %m1 = arith.constant 1 : i32
    aie.use_lock(%lkp, AcquireGreaterEqual, %m1)
    aie.dma_bd(%kb : memref<288xi8> offset = 0 len = 288)
    aie.use_lock(%lk, Release, %m1)
    aie.next_bd ^bd
  ^end:
    aie.end
  }
  %c = aie.core(%t) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c72 = arith.constant 72 : index
    %bz = arith.constant 0.0 : bf16
    %fz = arith.constant 0.0 : f32
    %one = arith.constant 1 : i32
    scf.for %blk = %c0 to %c16 step %c1 {
      aie.use_lock(%lk, AcquireGreaterEqual, %one)
      scf.for %i = %c0 to %c4 step %c1 {
        scf.for %j = %c0 to %c4 step %c1 {
          %qv = vector.transfer_read %q[%i, %c0], %bz {in_bounds = [true]} : memref<4x64xbf16>, vector<64xbf16>
          %qt = vector.shape_cast %qv : vector<64xbf16> to vector<8x8xbf16>
          %qm, %qe = aievec.to_bfp16 %qt : vector<8x8xbf16> -> vector<64xi8>, vector<8xi8>
          %ko = arith.muli %j, %c72 : index
          %km, %ke = aievec.load_bfp16 %kb[%ko] : memref<288xi8> -> vector<64xi8>, vector<8xi8>
          %av = vector.transfer_read %acc[%i, %j, %c0], %fz {in_bounds = [true]} : memref<4x4x64xf32>, vector<64xf32>
          %a = vector.shape_cast %av : vector<64xf32> to vector<8x8xf32>
          %r = aievec.matmul_aie2p_bfp16 %qm, %qe, %km, %ke, %a : vector<64xi8>, vector<8xi8>, vector<64xi8>, vector<8xi8> into vector<8x8xf32>
          %rv = vector.shape_cast %r : vector<8x8xf32> to vector<64xf32>
          vector.transfer_write %rv, %acc[%i, %j, %c0] {in_bounds = [true]} : vector<64xf32>, memref<4x4x64xf32>
          vector.transfer_write %qv, %q[%j, %c0] {in_bounds = [true]} : vector<64xbf16>, memref<4x64xbf16>
        }
      }
      aie.use_lock(%lkp, Release, %one)
    }
    aie.end
  }
}

// -----

// Each tile of q is converted only once, so there is nothing to gain.
// CHECK-LABEL: aie.device(npu2) @once
// CHECK-NOT: q_bfp16
aie.device(npu2) @once {
  %t = aie.tile(0, 2)
  %q = aie.buffer(%t) {sym_name = "q"} : memref<4x64xbf16>
  %kb = aie.buffer(%t) {sym_name = "kb"} : memref<288xi8>
  %acc = aie.buffer(%t) {sym_name = "acc"} : memref<4x4x64xf32>
  %lk = aie.lock(%t, 0) {init = 0 : i32}
  %lkp = aie.lock(%t, 1) {init = 1 : i32}
  %m = aie.mem(%t) {
    %0 = aie.dma_start(S2MM, 0, ^bd, ^end)
  ^bd:
    %m1 = arith.constant 1 : i32
    aie.use_lock(%lkp, AcquireGreaterEqual, %m1)
    aie.dma_bd(%kb : memref<288xi8> offset = 0 len = 288)
    aie.use_lock(%lk, Release, %m1)
    aie.next_bd ^bd
  ^end:
    aie.end
  }
  %c = aie.core(%t) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c72 = arith.constant 72 : index
    %bz = arith.constant 0.0 : bf16
    %fz = arith.constant 0.0 : f32
    %one = arith.constant 1 : i32
    scf.for %blk = %c0 to %c1 step %c1 {
      aie.use_lock(%lk, AcquireGreaterEqual, %one)
      scf.for %i = %c0 to %c4 step %c1 {
        scf.for %j = %c0 to %c1 step %c1 {
          %qv = vector.transfer_read %q[%i, %c0], %bz {in_bounds = [true]} : memref<4x64xbf16>, vector<64xbf16>
          %qt = vector.shape_cast %qv : vector<64xbf16> to vector<8x8xbf16>
          %qm, %qe = aievec.to_bfp16 %qt : vector<8x8xbf16> -> vector<64xi8>, vector<8xi8>
          %ko = arith.muli %j, %c72 : index
          %km, %ke = aievec.load_bfp16 %kb[%ko] : memref<288xi8> -> vector<64xi8>, vector<8xi8>
          %av = vector.transfer_read %acc[%i, %j, %c0], %fz {in_bounds = [true]} : memref<4x4x64xf32>, vector<64xf32>
          %a = vector.shape_cast %av : vector<64xf32> to vector<8x8xf32>
          %r = aievec.matmul_aie2p_bfp16 %qm, %qe, %km, %ke, %a : vector<64xi8>, vector<8xi8>, vector<64xi8>, vector<8xi8> into vector<8x8xf32>
          %rv = vector.shape_cast %r : vector<8x8xf32> to vector<64xf32>
          vector.transfer_write %rv, %acc[%i, %j, %c0] {in_bounds = [true]} : vector<64xf32>, memref<4x4x64xf32>
        }
      }
      aie.use_lock(%lkp, Release, %one)
    }
    aie.end
  }
}

// -----

// The copy does not fit in the core's memory next to the large buffer.
// CHECK-LABEL: aie.device(npu2) @no_room
// CHECK-NOT: q_bfp16
aie.device(npu2) @no_room {
  %t = aie.tile(0, 2)
  %q = aie.buffer(%t) {sym_name = "q"} : memref<4x64xbf16>
  %kb = aie.buffer(%t) {sym_name = "kb"} : memref<288xi8>
  %big = aie.buffer(%t) {sym_name = "big"} : memref<15500xf32>
  %acc = aie.buffer(%t) {sym_name = "acc"} : memref<4x4x64xf32>
  %lk = aie.lock(%t, 0) {init = 0 : i32}
  %lkp = aie.lock(%t, 1) {init = 1 : i32}
  %m = aie.mem(%t) {
    %0 = aie.dma_start(S2MM, 0, ^bd, ^end)
  ^bd:
    %m1 = arith.constant 1 : i32
    aie.use_lock(%lkp, AcquireGreaterEqual, %m1)
    aie.dma_bd(%kb : memref<288xi8> offset = 0 len = 288)
    aie.use_lock(%lk, Release, %m1)
    aie.next_bd ^bd
  ^end:
    aie.end
  }
  %c = aie.core(%t) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c72 = arith.constant 72 : index
    %bz = arith.constant 0.0 : bf16
    %fz = arith.constant 0.0 : f32
    %one = arith.constant 1 : i32
    scf.for %blk = %c0 to %c16 step %c1 {
      aie.use_lock(%lk, AcquireGreaterEqual, %one)
      scf.for %i = %c0 to %c4 step %c1 {
        scf.for %j = %c0 to %c4 step %c1 {
          %qv = vector.transfer_read %q[%i, %c0], %bz {in_bounds = [true]} : memref<4x64xbf16>, vector<64xbf16>
          %qt = vector.shape_cast %qv : vector<64xbf16> to vector<8x8xbf16>
          %qm, %qe = aievec.to_bfp16 %qt : vector<8x8xbf16> -> vector<64xi8>, vector<8xi8>
          %ko = arith.muli %j, %c72 : index
          %km, %ke = aievec.load_bfp16 %kb[%ko] : memref<288xi8> -> vector<64xi8>, vector<8xi8>
          %av = vector.transfer_read %acc[%i, %j, %c0], %fz {in_bounds = [true]} : memref<4x4x64xf32>, vector<64xf32>
          %a = vector.shape_cast %av : vector<64xf32> to vector<8x8xf32>
          %r = aievec.matmul_aie2p_bfp16 %qm, %qe, %km, %ke, %a : vector<64xi8>, vector<8xi8>, vector<64xi8>, vector<8xi8> into vector<8x8xf32>
          %rv = vector.shape_cast %r : vector<8x8xf32> to vector<64xf32>
          vector.transfer_write %rv, %acc[%i, %j, %c0] {in_bounds = [true]} : vector<64xf32>, memref<4x4x64xf32>
        }
      }
      aie.use_lock(%lkp, Release, %one)
    }
    aie.end
  }
}

// -----

// A DMA refills q in every iteration of the outer loop. The copy is built
// inside that loop, after the lock acquire, and serves the two inner loops,
// which convert each tile of q four times.
// CHECK-LABEL: aie.device(npu2) @refilled
// CHECK: aie.buffer({{.*}}) {sym_name = "q_bfp16"}
// CHECK: aie.core
// CHECK: scf.for
// CHECK-NEXT: aie.use_lock(%{{.*}}, AcquireGreaterEqual
// CHECK: aievec.fifo_start
// CHECK: aievec.fifo_st_push_bfp16
// CHECK: scf.for
// CHECK: aievec.load_bfp16 %q_bfp16
aie.device(npu2) @refilled {
  %t = aie.tile(0, 2)
  %q = aie.buffer(%t) {sym_name = "q"} : memref<4x64xbf16>
  %kb = aie.buffer(%t) {sym_name = "kb"} : memref<288xi8>
  %acc = aie.buffer(%t) {sym_name = "acc"} : memref<4x4x64xf32>
  %lq = aie.lock(%t, 0) {init = 0 : i32}
  %lqp = aie.lock(%t, 1) {init = 1 : i32}
  %m = aie.mem(%t) {
    %0 = aie.dma_start(S2MM, 0, ^bd, ^end)
  ^bd:
    %m1 = arith.constant 1 : i32
    aie.use_lock(%lqp, AcquireGreaterEqual, %m1)
    aie.dma_bd(%q : memref<4x64xbf16> offset = 0 len = 256)
    aie.use_lock(%lq, Release, %m1)
    aie.next_bd ^bd
  ^end:
    aie.end
  }
  %c = aie.core(%t) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c72 = arith.constant 72 : index
    %bz = arith.constant 0.0 : bf16
    %fz = arith.constant 0.0 : f32
    %one = arith.constant 1 : i32
    scf.for %blk = %c0 to %c16 step %c1 {
      aie.use_lock(%lq, AcquireGreaterEqual, %one)
      scf.for %i = %c0 to %c4 step %c1 {
        scf.for %j = %c0 to %c4 step %c1 {
          %qv = vector.transfer_read %q[%i, %c0], %bz {in_bounds = [true]} : memref<4x64xbf16>, vector<64xbf16>
          %qt = vector.shape_cast %qv : vector<64xbf16> to vector<8x8xbf16>
          %qm, %qe = aievec.to_bfp16 %qt : vector<8x8xbf16> -> vector<64xi8>, vector<8xi8>
          %ko = arith.muli %j, %c72 : index
          %km, %ke = aievec.load_bfp16 %kb[%ko] : memref<288xi8> -> vector<64xi8>, vector<8xi8>
          %av = vector.transfer_read %acc[%i, %j, %c0], %fz {in_bounds = [true]} : memref<4x4x64xf32>, vector<64xf32>
          %a = vector.shape_cast %av : vector<64xf32> to vector<8x8xf32>
          %r = aievec.matmul_aie2p_bfp16 %qm, %qe, %km, %ke, %a : vector<64xi8>, vector<8xi8>, vector<64xi8>, vector<8xi8> into vector<8x8xf32>
          %rv = vector.shape_cast %r : vector<8x8xf32> to vector<64xf32>
          vector.transfer_write %rv, %acc[%i, %j, %c0] {in_bounds = [true]} : vector<64xf32>, memref<4x4x64xf32>
        }
      }
      aie.use_lock(%lqp, Release, %one)
    }
    aie.end
  }
}

// -----

// The innermost loop advances a pointer through q by one tile per iteration
// and runs twice per iteration of the outer loop. The copy stores the blocks
// in the order the pointer visits them, so the innermost loop reads them as
// a load stream alongside the stream of k.
// CHECK-LABEL: aie.device(npu2) @stepped
// CHECK: aie.core
// CHECK: %[[Q:.*]], %[[QS:.*]], %[[QN:.*]] = aievec.fifo_start %q_bfp16[%c0]
// CHECK-COUNT-4: aievec.fifo_st_push_bfp16
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for {{.*}} iter_args({{.*}}%[[A:arg[0-9]+]] = %[[Q]], %[[S:arg[0-9]+]] = %[[QS]], %[[N:arg[0-9]+]] = %[[QN]]
// CHECK: aievec.fifo_ld_pop_bfp16 %[[A]], %[[S]], %[[N]]
// CHECK: aievec.fifo_ld_pop_bfp16
// CHECK: aievec.matmul_aie2p_bfp16
// CHECK-NOT: aievec.load_bfp16
aie.device(npu2) @stepped {
  %t = aie.tile(0, 2)
  %q = aie.buffer(%t) {sym_name = "q"} : memref<256xbf16>
  %kb = aie.buffer(%t) {sym_name = "kb"} : memref<288xi8>
  %acc = aie.buffer(%t) {sym_name = "acc"} : memref<64xf32>
  %c = aie.core(%t) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index
    %c72 = arith.constant 72 : index
    %bz = arith.constant 0.0 : bf16
    %fz = arith.constant 0.0 : f32
    scf.for %i = %c0 to %c4 step %c1 {
      scf.for %pass = %c0 to %c2 step %c1 {
        %av = vector.transfer_read %acc[%c0], %fz {in_bounds = [true]} : memref<64xf32>, vector<64xf32>
        %a0 = vector.shape_cast %av : vector<64xf32> to vector<8x8xf32>
        %res:3 = scf.for %j = %c0 to %c4 step %c1 iter_args(%p = %c0, %ko = %c0, %a = %a0) -> (index, index, vector<8x8xf32>) {
          %qv = vector.transfer_read %q[%p], %bz {in_bounds = [true]} : memref<256xbf16>, vector<64xbf16>
          %qt = vector.shape_cast %qv : vector<64xbf16> to vector<8x8xbf16>
          %qm, %qe = aievec.to_bfp16 %qt : vector<8x8xbf16> -> vector<64xi8>, vector<8xi8>
          %km, %ke = aievec.load_bfp16 %kb[%ko] : memref<288xi8> -> vector<64xi8>, vector<8xi8>
          %r = aievec.matmul_aie2p_bfp16 %qm, %qe, %km, %ke, %a : vector<64xi8>, vector<8xi8>, vector<64xi8>, vector<8xi8> into vector<8x8xf32>
          %pn = arith.addi %p, %c64 : index
          %kn = arith.addi %ko, %c72 : index
          scf.yield %pn, %kn, %r : index, index, vector<8x8xf32>
        }
        %rv = vector.shape_cast %res#2 : vector<8x8xf32> to vector<64xf32>
        vector.transfer_write %rv, %acc[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32>
      }
    }
    aie.end
  }
}

// -----

// Each tile of q is converted four times, but the reads start 4 values into
// a tile, so they are not reads of whole tiles and nothing is hoisted.
// CHECK-LABEL: aie.device(npu2) @misaligned
// CHECK-NOT: q_bfp16
aie.device(npu2) @misaligned {
  %t = aie.tile(0, 2)
  %q = aie.buffer(%t) {sym_name = "q"} : memref<320xbf16>
  %kb = aie.buffer(%t) {sym_name = "kb"} : memref<288xi8>
  %acc = aie.buffer(%t) {sym_name = "acc"} : memref<4x4x64xf32>
  %lk = aie.lock(%t, 0) {init = 0 : i32}
  %lkp = aie.lock(%t, 1) {init = 1 : i32}
  %m = aie.mem(%t) {
    %0 = aie.dma_start(S2MM, 0, ^bd, ^end)
  ^bd:
    %m1 = arith.constant 1 : i32
    aie.use_lock(%lkp, AcquireGreaterEqual, %m1)
    aie.dma_bd(%kb : memref<288xi8> offset = 0 len = 288)
    aie.use_lock(%lk, Release, %m1)
    aie.next_bd ^bd
  ^end:
    aie.end
  }
  %c = aie.core(%t) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %c72 = arith.constant 72 : index
    %bz = arith.constant 0.0 : bf16
    %fz = arith.constant 0.0 : f32
    %one = arith.constant 1 : i32
    scf.for %blk = %c0 to %c1 step %c1 {
      aie.use_lock(%lk, AcquireGreaterEqual, %one)
      scf.for %i = %c0 to %c4 step %c1 {
        scf.for %j = %c0 to %c4 step %c1 {
          %qi = arith.muli %i, %c64 : index
          %qo = arith.addi %qi, %c4 : index
          %qv = vector.transfer_read %q[%qo], %bz {in_bounds = [true]} : memref<320xbf16>, vector<64xbf16>
          %qt = vector.shape_cast %qv : vector<64xbf16> to vector<8x8xbf16>
          %qm, %qe = aievec.to_bfp16 %qt : vector<8x8xbf16> -> vector<64xi8>, vector<8xi8>
          %ko = arith.muli %j, %c72 : index
          %km, %ke = aievec.load_bfp16 %kb[%ko] : memref<288xi8> -> vector<64xi8>, vector<8xi8>
          %av = vector.transfer_read %acc[%i, %j, %c0], %fz {in_bounds = [true]} : memref<4x4x64xf32>, vector<64xf32>
          %a = vector.shape_cast %av : vector<64xf32> to vector<8x8xf32>
          %r = aievec.matmul_aie2p_bfp16 %qm, %qe, %km, %ke, %a : vector<64xi8>, vector<8xi8>, vector<64xi8>, vector<8xi8> into vector<8x8xf32>
          %rv = vector.shape_cast %r : vector<8x8xf32> to vector<64xf32>
          vector.transfer_write %rv, %acc[%i, %j, %c0] {in_bounds = [true]} : vector<64xf32>, memref<4x4x64xf32>
        }
      }
      aie.use_lock(%lkp, Release, %one)
    }
    aie.end
  }
}

// -----

// The inner loop's trip count is only known at run time. Such a loop is
// assumed to repeat, so each tile of q counts as converted twice and q is
// converted once before the loops.
// CHECK-LABEL: aie.device(npu2) @runtime_trips
// CHECK: sym_name = "q_bfp16"
// CHECK: aie.core
// CHECK: aievec.fifo_st_push_bfp16
// CHECK: scf.for
aie.device(npu2) @runtime_trips {
  %t = aie.tile(0, 2)
  %q = aie.buffer(%t) {sym_name = "q"} : memref<4x64xbf16>
  %kb = aie.buffer(%t) {sym_name = "kb"} : memref<288xi8>
  %acc = aie.buffer(%t) {sym_name = "acc"} : memref<4x4x64xf32>
  %rtp = aie.buffer(%t) {sym_name = "rtp"} : memref<1xi32>
  %lk = aie.lock(%t, 0) {init = 0 : i32}
  %lkp = aie.lock(%t, 1) {init = 1 : i32}
  %m = aie.mem(%t) {
    %0 = aie.dma_start(S2MM, 0, ^bd, ^end)
  ^bd:
    %m1 = arith.constant 1 : i32
    aie.use_lock(%lkp, AcquireGreaterEqual, %m1)
    aie.dma_bd(%kb : memref<288xi8> offset = 0 len = 288)
    aie.use_lock(%lk, Release, %m1)
    aie.next_bd ^bd
  ^end:
    aie.end
  }
  %c = aie.core(%t) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c72 = arith.constant 72 : index
    %bz = arith.constant 0.0 : bf16
    %fz = arith.constant 0.0 : f32
    %one = arith.constant 1 : i32
    %n32 = memref.load %rtp[%c0] : memref<1xi32>
    %n = arith.index_cast %n32 : i32 to index
    scf.for %blk = %c0 to %c1 step %c1 {
      aie.use_lock(%lk, AcquireGreaterEqual, %one)
      scf.for %i = %c0 to %c4 step %c1 {
        scf.for %j = %c0 to %n step %c1 {
          %qv = vector.transfer_read %q[%i, %c0], %bz {in_bounds = [true]} : memref<4x64xbf16>, vector<64xbf16>
          %qt = vector.shape_cast %qv : vector<64xbf16> to vector<8x8xbf16>
          %qm, %qe = aievec.to_bfp16 %qt : vector<8x8xbf16> -> vector<64xi8>, vector<8xi8>
          %ko = arith.muli %j, %c72 : index
          %km, %ke = aievec.load_bfp16 %kb[%ko] : memref<288xi8> -> vector<64xi8>, vector<8xi8>
          %av = vector.transfer_read %acc[%i, %j, %c0], %fz {in_bounds = [true]} : memref<4x4x64xf32>, vector<64xf32>
          %a = vector.shape_cast %av : vector<64xf32> to vector<8x8xf32>
          %r = aievec.matmul_aie2p_bfp16 %qm, %qe, %km, %ke, %a : vector<64xi8>, vector<8xi8>, vector<64xi8>, vector<8xi8> into vector<8x8xf32>
          %rv = vector.shape_cast %r : vector<8x8xf32> to vector<64xf32>
          vector.transfer_write %rv, %acc[%i, %j, %c0] {in_bounds = [true]} : vector<64xf32>, memref<4x4x64xf32>
        }
      }
      aie.use_lock(%lkp, Release, %one)
    }
    aie.end
  }
}

// -----

// q's tile is read once before the loops and converted inside them. A copy
// made before the loops would be read before it is written, so the
// conversion stays.
// CHECK-LABEL: aie.device(npu2) @read_outside
// CHECK-NOT: q_bfp16
// CHECK: scf.for
// CHECK: aievec.to_bfp16
aie.device(npu2) @read_outside {
  %t = aie.tile(0, 2)
  %q = aie.buffer(%t) {sym_name = "q"} : memref<4x64xbf16>
  %kb = aie.buffer(%t) {sym_name = "kb"} : memref<288xi8>
  %acc = aie.buffer(%t) {sym_name = "acc"} : memref<4x4x64xf32>
  %lk = aie.lock(%t, 0) {init = 0 : i32}
  %lkp = aie.lock(%t, 1) {init = 1 : i32}
  %m = aie.mem(%t) {
    %0 = aie.dma_start(S2MM, 0, ^bd, ^end)
  ^bd:
    %m1 = arith.constant 1 : i32
    aie.use_lock(%lkp, AcquireGreaterEqual, %m1)
    aie.dma_bd(%kb : memref<288xi8> offset = 0 len = 288)
    aie.use_lock(%lk, Release, %m1)
    aie.next_bd ^bd
  ^end:
    aie.end
  }
  %c = aie.core(%t) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c72 = arith.constant 72 : index
    %bz = arith.constant 0.0 : bf16
    %fz = arith.constant 0.0 : f32
    %one = arith.constant 1 : i32
    %qv = vector.transfer_read %q[%c0, %c0], %bz {in_bounds = [true]} : memref<4x64xbf16>, vector<64xbf16>
    %qt = vector.shape_cast %qv : vector<64xbf16> to vector<8x8xbf16>
    scf.for %blk = %c0 to %c16 step %c1 {
      aie.use_lock(%lk, AcquireGreaterEqual, %one)
      scf.for %i = %c0 to %c4 step %c1 {
        scf.for %j = %c0 to %c4 step %c1 {
          %qm, %qe = aievec.to_bfp16 %qt : vector<8x8xbf16> -> vector<64xi8>, vector<8xi8>
          %ko = arith.muli %j, %c72 : index
          %km, %ke = aievec.load_bfp16 %kb[%ko] : memref<288xi8> -> vector<64xi8>, vector<8xi8>
          %av = vector.transfer_read %acc[%i, %j, %c0], %fz {in_bounds = [true]} : memref<4x4x64xf32>, vector<64xf32>
          %a = vector.shape_cast %av : vector<64xf32> to vector<8x8xf32>
          %r = aievec.matmul_aie2p_bfp16 %qm, %qe, %km, %ke, %a : vector<64xi8>, vector<8xi8>, vector<64xi8>, vector<8xi8> into vector<8x8xf32>
          %rv = vector.shape_cast %r : vector<8x8xf32> to vector<64xf32>
          vector.transfer_write %rv, %acc[%i, %j, %c0] {in_bounds = [true]} : vector<64xf32>, memref<4x4x64xf32>
        }
      }
      aie.use_lock(%lkp, Release, %one)
    }
    aie.end
  }
}

// -----

// The name the copy would take is in use, so the copy gets another.
// CHECK-LABEL: aie.device(npu2) @name_taken
// CHECK: sym_name = "q_bfp16_0"
aie.device(npu2) @name_taken {
  %t = aie.tile(0, 2)
  %q = aie.buffer(%t) {sym_name = "q"} : memref<4x64xbf16>
  %kb = aie.buffer(%t) {sym_name = "kb"} : memref<288xi8>
  %taken = aie.buffer(%t) {sym_name = "q_bfp16"} : memref<4xi8>
  %acc = aie.buffer(%t) {sym_name = "acc"} : memref<4x4x64xf32>
  %lk = aie.lock(%t, 0) {init = 0 : i32}
  %lkp = aie.lock(%t, 1) {init = 1 : i32}
  %m = aie.mem(%t) {
    %0 = aie.dma_start(S2MM, 0, ^bd, ^end)
  ^bd:
    %m1 = arith.constant 1 : i32
    aie.use_lock(%lkp, AcquireGreaterEqual, %m1)
    aie.dma_bd(%kb : memref<288xi8> offset = 0 len = 288)
    aie.use_lock(%lk, Release, %m1)
    aie.next_bd ^bd
  ^end:
    aie.end
  }
  %c = aie.core(%t) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c72 = arith.constant 72 : index
    %bz = arith.constant 0.0 : bf16
    %fz = arith.constant 0.0 : f32
    %one = arith.constant 1 : i32
    scf.for %blk = %c0 to %c16 step %c1 {
      aie.use_lock(%lk, AcquireGreaterEqual, %one)
      scf.for %i = %c0 to %c4 step %c1 {
        scf.for %j = %c0 to %c4 step %c1 {
          %qv = vector.transfer_read %q[%i, %c0], %bz {in_bounds = [true]} : memref<4x64xbf16>, vector<64xbf16>
          %qt = vector.shape_cast %qv : vector<64xbf16> to vector<8x8xbf16>
          %qm, %qe = aievec.to_bfp16 %qt : vector<8x8xbf16> -> vector<64xi8>, vector<8xi8>
          %ko = arith.muli %j, %c72 : index
          %km, %ke = aievec.load_bfp16 %kb[%ko] : memref<288xi8> -> vector<64xi8>, vector<8xi8>
          %av = vector.transfer_read %acc[%i, %j, %c0], %fz {in_bounds = [true]} : memref<4x4x64xf32>, vector<64xf32>
          %a = vector.shape_cast %av : vector<64xf32> to vector<8x8xf32>
          %r = aievec.matmul_aie2p_bfp16 %qm, %qe, %km, %ke, %a : vector<64xi8>, vector<8xi8>, vector<64xi8>, vector<8xi8> into vector<8x8xf32>
          %rv = vector.shape_cast %r : vector<8x8xf32> to vector<64xf32>
          vector.transfer_write %rv, %acc[%i, %j, %c0] {in_bounds = [true]} : vector<64xf32>, memref<4x4x64xf32>
        }
      }
      aie.use_lock(%lkp, Release, %one)
    }
    aie.end
  }
}

// -----

// Another core writes q under locks no DMA of q uses. Any lock may then
// guard q, so the copy is built after the lock is acquired, in the loop.
// CHECK-LABEL: aie.device(npu2) @other_core
// CHECK: aie.core(%tile_0_2) {
// CHECK-NOT: aievec.fifo_st_push_bfp16
// CHECK: scf.for
// CHECK: aie.use_lock(%{{.*}}, AcquireGreaterEqual
// CHECK: aievec.fifo_st_push_bfp16
aie.device(npu2) @other_core {
  %t = aie.tile(0, 2)
  %t3 = aie.tile(0, 3)
  %q = aie.buffer(%t) {sym_name = "q"} : memref<4x64xbf16>
  %kb = aie.buffer(%t) {sym_name = "kb"} : memref<288xi8>
  %acc = aie.buffer(%t) {sym_name = "acc"} : memref<4x4x64xf32>
  %lq = aie.lock(%t, 2) {init = 0 : i32}
  %lqp = aie.lock(%t, 3) {init = 1 : i32}
  // Producer core writes q through shared memory, no DMA.
  %c3 = aie.core(%t3) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %one = arith.constant 1 : i32
    %v = arith.constant dense<1.0> : vector<64xbf16>
    scf.for %blk = %c0 to %c16 step %c1 {
      aie.use_lock(%lqp, AcquireGreaterEqual, %one)
      vector.transfer_write %v, %q[%c0, %c0] {in_bounds = [true]} : vector<64xbf16>, memref<4x64xbf16>
      aie.use_lock(%lq, Release, %one)
    }
    aie.end
  }
  %c = aie.core(%t) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c72 = arith.constant 72 : index
    %bz = arith.constant 0.0 : bf16
    %fz = arith.constant 0.0 : f32
    %one = arith.constant 1 : i32
    scf.for %blk = %c0 to %c16 step %c1 {
      aie.use_lock(%lq, AcquireGreaterEqual, %one)
      scf.for %i = %c0 to %c4 step %c1 {
        scf.for %j = %c0 to %c4 step %c1 {
          %qv = vector.transfer_read %q[%i, %c0], %bz {in_bounds = [true]} : memref<4x64xbf16>, vector<64xbf16>
          %qt = vector.shape_cast %qv : vector<64xbf16> to vector<8x8xbf16>
          %qm, %qe = aievec.to_bfp16 %qt : vector<8x8xbf16> -> vector<64xi8>, vector<8xi8>
          %ko = arith.muli %j, %c72 : index
          %km, %ke = aievec.load_bfp16 %kb[%ko] : memref<288xi8> -> vector<64xi8>, vector<8xi8>
          %av = vector.transfer_read %acc[%i, %j, %c0], %fz {in_bounds = [true]} : memref<4x4x64xf32>, vector<64xf32>
          %a = vector.shape_cast %av : vector<64xf32> to vector<8x8xf32>
          %r = aievec.matmul_aie2p_bfp16 %qm, %qe, %km, %ke, %a : vector<64xi8>, vector<8xi8>, vector<64xi8>, vector<8xi8> into vector<8x8xf32>
          %rv = vector.shape_cast %r : vector<8x8xf32> to vector<64xf32>
          vector.transfer_write %rv, %acc[%i, %j, %c0] {in_bounds = [true]} : vector<64xf32>, memref<4x4x64xf32>
        }
      }
      aie.use_lock(%lqp, Release, %one)
    }
    aie.end
  }
}

// -----

// q's DMA chain releases the lock that guards q on the block after q's own.
// The copy is built after that lock is acquired, in the loop.
// CHECK-LABEL: aie.device(npu2) @bd_chain
// CHECK: aie.core(%tile_0_2) {
// CHECK-NOT: aievec.fifo_st_push_bfp16
// CHECK: scf.for
// CHECK: aie.use_lock(%{{.*}}, AcquireGreaterEqual
// CHECK: aievec.fifo_st_push_bfp16
aie.device(npu2) @bd_chain {
  %t = aie.tile(0, 2)
    %q = aie.buffer(%t) {sym_name = "q"} : memref<4x64xbf16>
  %kb = aie.buffer(%t) {sym_name = "kb"} : memref<288xi8>
  %acc = aie.buffer(%t) {sym_name = "acc"} : memref<4x4x64xf32>
  %lq = aie.lock(%t, 2) {init = 0 : i32}
  %lqp = aie.lock(%t, 3) {init = 1 : i32}
  %other = aie.buffer(%t) {sym_name = "other"} : memref<64xbf16>
  %mm = aie.mem(%t) {
    %0 = aie.dma_start(S2MM, 0, ^bd0, ^end)
  ^bd0:
    %m1 = arith.constant 1 : i32
    aie.use_lock(%lqp, AcquireGreaterEqual, %m1)
    aie.dma_bd(%q : memref<4x64xbf16> offset = 0 len = 256)
    aie.next_bd ^bd1
  ^bd1:
    %m2 = arith.constant 1 : i32
    aie.dma_bd(%other : memref<64xbf16> offset = 0 len = 64)
    aie.use_lock(%lq, Release, %m2)
    aie.next_bd ^bd0
  ^end:
    aie.end
  }
  %c = aie.core(%t) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c72 = arith.constant 72 : index
    %bz = arith.constant 0.0 : bf16
    %fz = arith.constant 0.0 : f32
    %one = arith.constant 1 : i32
    scf.for %blk = %c0 to %c16 step %c1 {
      aie.use_lock(%lq, AcquireGreaterEqual, %one)
      scf.for %i = %c0 to %c4 step %c1 {
        scf.for %j = %c0 to %c4 step %c1 {
          %qv = vector.transfer_read %q[%i, %c0], %bz {in_bounds = [true]} : memref<4x64xbf16>, vector<64xbf16>
          %qt = vector.shape_cast %qv : vector<64xbf16> to vector<8x8xbf16>
          %qm, %qe = aievec.to_bfp16 %qt : vector<8x8xbf16> -> vector<64xi8>, vector<8xi8>
          %ko = arith.muli %j, %c72 : index
          %km, %ke = aievec.load_bfp16 %kb[%ko] : memref<288xi8> -> vector<64xi8>, vector<8xi8>
          %av = vector.transfer_read %acc[%i, %j, %c0], %fz {in_bounds = [true]} : memref<4x4x64xf32>, vector<64xf32>
          %a = vector.shape_cast %av : vector<64xf32> to vector<8x8xf32>
          %r = aievec.matmul_aie2p_bfp16 %qm, %qe, %km, %ke, %a : vector<64xi8>, vector<8xi8>, vector<64xi8>, vector<8xi8> into vector<8x8xf32>
          %rv = vector.shape_cast %r : vector<8x8xf32> to vector<64xf32>
          vector.transfer_write %rv, %acc[%i, %j, %c0] {in_bounds = [true]} : vector<64xf32>, memref<4x4x64xf32>
        }
      }
      aie.use_lock(%lqp, Release, %one)
    }
    aie.end
  }
}
