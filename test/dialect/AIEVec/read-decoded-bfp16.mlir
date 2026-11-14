//===- read-decoded-bfp16.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file --optimize-aievec="aie-target=aie2p" | FileCheck %s

// A DMA brings k as 4 bfp16ebs8 blocks, which the core decodes into bf16
// tiles that are only read whole by untransposed conversions. The matmuls
// load the blocks instead (here as a load stream), the decode loop and the
// bf16 tile buffer are removed, and the lock release of the block buffer
// moves after the reads.
// CHECK-LABEL: aie.device(npu2) @decoded
// CHECK-NOT: sym_name = "k"
// CHECK: aie.core
// CHECK-NOT: arith.sitofp
// CHECK: aie.use_lock(%{{.*}}, AcquireGreaterEqual
// CHECK: aievec.fifo_start
// CHECK: scf.for
// CHECK: aievec.fifo_ld_pop_bfp16
// CHECK: aievec.matmul_aie2p_bfp16
// CHECK: aie.use_lock(%{{.*}}, Release
aie.device(npu2) @decoded {
  %t = aie.tile(0, 2)
  %kb = aie.buffer(%t) {sym_name = "kb"} : memref<4x8x9xi8>
  %k = aie.buffer(%t) {sym_name = "k"} : memref<4x8x8xbf16>
  %q = aie.buffer(%t) {sym_name = "q"} : memref<64xbf16>
  %acc = aie.buffer(%t) {sym_name = "acc"} : memref<4x64xf32>
  %lk = aie.lock(%t, 0) {init = 0 : i32}
  %lkp = aie.lock(%t, 1) {init = 1 : i32}
  %m = aie.mem(%t) {
    %0 = aie.dma_start(S2MM, 0, ^bd, ^end)
  ^bd:
    %m1 = arith.constant 1 : i32
    aie.use_lock(%lkp, AcquireGreaterEqual, %m1)
    aie.dma_bd(%kb : memref<4x8x9xi8> offset = 0 len = 288)
    aie.use_lock(%lk, Release, %m1)
    aie.next_bd ^bd
  ^end:
    aie.end
  }
  %c = aie.core(%t) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c6 = arith.constant 6 : i32
    %c23 = arith.constant 23 : i32
    %bz = arith.constant 0.0 : bf16
    %fz = arith.constant 0.0 : f32
    %one = arith.constant 1 : i32
    scf.for %blk = %c0 to %c16 step %c1 {
      aie.use_lock(%lk, AcquireGreaterEqual, %one)
      scf.for %i = %c0 to %c4 step %c1 {
        scf.for %r = %c0 to %c8 step %c1 {
          scf.for %cc = %c0 to %c8 step %c1 {
            %e8 = memref.load %kb[%i, %r, %c0] : memref<4x8x9xi8>
            %c1x = arith.addi %cc, %c1 : index
            %m8 = memref.load %kb[%i, %r, %c1x] : memref<4x8x9xi8>
            %ez = arith.extui %e8 : i8 to i32
            %eb = arith.subi %ez, %c6 : i32
            %es = arith.shli %eb, %c23 : i32
            %sc = arith.bitcast %es : i32 to f32
            %mf = arith.sitofp %m8 : i8 to f32
            %v = arith.mulf %mf, %sc : f32
            %vb = arith.truncf %v : f32 to bf16
            memref.store %vb, %k[%i, %r, %cc] : memref<4x8x8xbf16>
          }
        }
      }
      aie.use_lock(%lkp, Release, %one)
      %qv = vector.transfer_read %q[%c0], %bz {in_bounds = [true]} : memref<64xbf16>, vector<64xbf16>
      %qt = vector.shape_cast %qv : vector<64xbf16> to vector<8x8xbf16>
      %qm, %qe = aievec.to_bfp16 %qt : vector<8x8xbf16> -> vector<64xi8>, vector<8xi8>
      scf.for %j = %c0 to %c4 step %c1 {
        %kv = vector.transfer_read %k[%j, %c0, %c0], %bz {in_bounds = [true, true]} : memref<4x8x8xbf16>, vector<8x8xbf16>
        %km, %ke = aievec.to_bfp16 %kv : vector<8x8xbf16> -> vector<64xi8>, vector<8xi8>
        %av = vector.transfer_read %acc[%j, %c0], %fz {in_bounds = [true]} : memref<4x64xf32>, vector<64xf32>
        %a = vector.shape_cast %av : vector<64xf32> to vector<8x8xf32>
        %res = aievec.matmul_aie2p_bfp16 %qm, %qe, %km, %ke, %a : vector<64xi8>, vector<8xi8>, vector<64xi8>, vector<8xi8> into vector<8x8xf32>
        %rv = vector.shape_cast %res : vector<8x8xf32> to vector<64xf32>
        vector.transfer_write %rv, %acc[%j, %c0] {in_bounds = [true]} : vector<64xf32>, memref<4x64xf32>
      }
    }
    aie.end
  }
}

// -----

// The tiles are converted transposed, which the blocks cannot provide, so
// the decode stays.
// CHECK-LABEL: aie.device(npu2) @transposed
// CHECK: arith.sitofp
// CHECK-NOT: aievec.load_bfp16
aie.device(npu2) @transposed {
  %t = aie.tile(0, 2)
  %kb = aie.buffer(%t) {sym_name = "kb"} : memref<4x8x9xi8>
  %k = aie.buffer(%t) {sym_name = "k"} : memref<4x8x8xbf16>
  %q = aie.buffer(%t) {sym_name = "q"} : memref<64xbf16>
  %acc = aie.buffer(%t) {sym_name = "acc"} : memref<4x64xf32>
  %lk = aie.lock(%t, 0) {init = 0 : i32}
  %lkp = aie.lock(%t, 1) {init = 1 : i32}
  %m = aie.mem(%t) {
    %0 = aie.dma_start(S2MM, 0, ^bd, ^end)
  ^bd:
    %m1 = arith.constant 1 : i32
    aie.use_lock(%lkp, AcquireGreaterEqual, %m1)
    aie.dma_bd(%kb : memref<4x8x9xi8> offset = 0 len = 288)
    aie.use_lock(%lk, Release, %m1)
    aie.next_bd ^bd
  ^end:
    aie.end
  }
  %c = aie.core(%t) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c6 = arith.constant 6 : i32
    %c23 = arith.constant 23 : i32
    %bz = arith.constant 0.0 : bf16
    %fz = arith.constant 0.0 : f32
    %one = arith.constant 1 : i32
    scf.for %blk = %c0 to %c16 step %c1 {
      aie.use_lock(%lk, AcquireGreaterEqual, %one)
      scf.for %i = %c0 to %c4 step %c1 {
        scf.for %r = %c0 to %c8 step %c1 {
          scf.for %cc = %c0 to %c8 step %c1 {
            %e8 = memref.load %kb[%i, %r, %c0] : memref<4x8x9xi8>
            %c1x = arith.addi %cc, %c1 : index
            %m8 = memref.load %kb[%i, %r, %c1x] : memref<4x8x9xi8>
            %ez = arith.extui %e8 : i8 to i32
            %eb = arith.subi %ez, %c6 : i32
            %es = arith.shli %eb, %c23 : i32
            %sc = arith.bitcast %es : i32 to f32
            %mf = arith.sitofp %m8 : i8 to f32
            %v = arith.mulf %mf, %sc : f32
            %vb = arith.truncf %v : f32 to bf16
            memref.store %vb, %k[%i, %r, %cc] : memref<4x8x8xbf16>
          }
        }
      }
      aie.use_lock(%lkp, Release, %one)
      %qv = vector.transfer_read %q[%c0], %bz {in_bounds = [true]} : memref<64xbf16>, vector<64xbf16>
      %qt = vector.shape_cast %qv : vector<64xbf16> to vector<8x8xbf16>
      %qm, %qe = aievec.to_bfp16 %qt : vector<8x8xbf16> -> vector<64xi8>, vector<8xi8>
      scf.for %j = %c0 to %c4 step %c1 {
        %kv = vector.transfer_read %k[%j, %c0, %c0], %bz {in_bounds = [true, true]} : memref<4x8x8xbf16>, vector<8x8xbf16>
        %km, %ke = aievec.to_bfp16 %kv transpose : vector<8x8xbf16> -> vector<64xi8>, vector<8xi8>
        %av = vector.transfer_read %acc[%j, %c0], %fz {in_bounds = [true]} : memref<4x64xf32>, vector<64xf32>
        %a = vector.shape_cast %av : vector<64xf32> to vector<8x8xf32>
        %res = aievec.matmul_aie2p_bfp16 %qm, %qe, %km, %ke, %a : vector<64xi8>, vector<8xi8>, vector<64xi8>, vector<8xi8> into vector<8x8xf32>
        %rv = vector.shape_cast %res : vector<8x8xf32> to vector<64xf32>
        vector.transfer_write %rv, %acc[%j, %c0] {in_bounds = [true]} : vector<64xf32>, memref<4x64xf32>
      }
    }
    aie.end
  }
}

// -----

// The tiles are also used as bf16 values, so the decode stays.
// CHECK-LABEL: aie.device(npu2) @other_use
// CHECK: arith.sitofp
// CHECK-NOT: aievec.load_bfp16
aie.device(npu2) @other_use {
  %t = aie.tile(0, 2)
  %kb = aie.buffer(%t) {sym_name = "kb"} : memref<4x8x9xi8>
  %k = aie.buffer(%t) {sym_name = "k"} : memref<4x8x8xbf16>
  %q = aie.buffer(%t) {sym_name = "q"} : memref<64xbf16>
  %acc = aie.buffer(%t) {sym_name = "acc"} : memref<4x64xf32>
  %lk = aie.lock(%t, 0) {init = 0 : i32}
  %lkp = aie.lock(%t, 1) {init = 1 : i32}
  %m = aie.mem(%t) {
    %0 = aie.dma_start(S2MM, 0, ^bd, ^end)
  ^bd:
    %m1 = arith.constant 1 : i32
    aie.use_lock(%lkp, AcquireGreaterEqual, %m1)
    aie.dma_bd(%kb : memref<4x8x9xi8> offset = 0 len = 288)
    aie.use_lock(%lk, Release, %m1)
    aie.next_bd ^bd
  ^end:
    aie.end
  }
  %c = aie.core(%t) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c6 = arith.constant 6 : i32
    %c23 = arith.constant 23 : i32
    %bz = arith.constant 0.0 : bf16
    %fz = arith.constant 0.0 : f32
    %one = arith.constant 1 : i32
    scf.for %blk = %c0 to %c16 step %c1 {
      aie.use_lock(%lk, AcquireGreaterEqual, %one)
      scf.for %i = %c0 to %c4 step %c1 {
        scf.for %r = %c0 to %c8 step %c1 {
          scf.for %cc = %c0 to %c8 step %c1 {
            %e8 = memref.load %kb[%i, %r, %c0] : memref<4x8x9xi8>
            %c1x = arith.addi %cc, %c1 : index
            %m8 = memref.load %kb[%i, %r, %c1x] : memref<4x8x9xi8>
            %ez = arith.extui %e8 : i8 to i32
            %eb = arith.subi %ez, %c6 : i32
            %es = arith.shli %eb, %c23 : i32
            %sc = arith.bitcast %es : i32 to f32
            %mf = arith.sitofp %m8 : i8 to f32
            %v = arith.mulf %mf, %sc : f32
            %vb = arith.truncf %v : f32 to bf16
            memref.store %vb, %k[%i, %r, %cc] : memref<4x8x8xbf16>
          }
        }
      }
      aie.use_lock(%lkp, Release, %one)
      %qv = vector.transfer_read %q[%c0], %bz {in_bounds = [true]} : memref<64xbf16>, vector<64xbf16>
      %qt = vector.shape_cast %qv : vector<64xbf16> to vector<8x8xbf16>
      %qm, %qe = aievec.to_bfp16 %qt : vector<8x8xbf16> -> vector<64xi8>, vector<8xi8>
      scf.for %j = %c0 to %c4 step %c1 {
        %kv = vector.transfer_read %k[%j, %c0, %c0], %bz {in_bounds = [true, true]} : memref<4x8x8xbf16>, vector<8x8xbf16>
        %km, %ke = aievec.to_bfp16 %kv : vector<8x8xbf16> -> vector<64xi8>, vector<8xi8>
        %av = vector.transfer_read %acc[%j, %c0], %fz {in_bounds = [true]} : memref<4x64xf32>, vector<64xf32>
        %a = vector.shape_cast %av : vector<64xf32> to vector<8x8xf32>
        %res = aievec.matmul_aie2p_bfp16 %qm, %qe, %km, %ke, %a : vector<64xi8>, vector<8xi8>, vector<64xi8>, vector<8xi8> into vector<8x8xf32>
        %rv = vector.shape_cast %res : vector<8x8xf32> to vector<64xf32>
        vector.transfer_write %rv, %acc[%j, %c0] {in_bounds = [true]} : vector<64xf32>, memref<4x64xf32>
        %kf = vector.shape_cast %kv : vector<8x8xbf16> to vector<64xbf16>
        vector.transfer_write %kf, %q[%c0] {in_bounds = [true]} : vector<64xbf16>, memref<64xbf16>
      }
    }
    aie.end
  }
}

// -----

// The block buffer's lock is acquired again between the decode and the
// reads, so the release cannot move past it and the decode stays.
// CHECK-LABEL: aie.device(npu2) @lock_between
// CHECK: arith.sitofp
// CHECK-NOT: aievec.load_bfp16
aie.device(npu2) @lock_between {
  %t = aie.tile(0, 2)
  %kb = aie.buffer(%t) {sym_name = "kb"} : memref<4x8x9xi8>
  %k = aie.buffer(%t) {sym_name = "k"} : memref<4x8x8xbf16>
  %q = aie.buffer(%t) {sym_name = "q"} : memref<64xbf16>
  %acc = aie.buffer(%t) {sym_name = "acc"} : memref<4x64xf32>
  %lk = aie.lock(%t, 0) {init = 0 : i32}
  %lkp = aie.lock(%t, 1) {init = 1 : i32}
  %m = aie.mem(%t) {
    %0 = aie.dma_start(S2MM, 0, ^bd, ^end)
  ^bd:
    %m1 = arith.constant 1 : i32
    aie.use_lock(%lkp, AcquireGreaterEqual, %m1)
    aie.dma_bd(%kb : memref<4x8x9xi8> offset = 0 len = 288)
    aie.use_lock(%lk, Release, %m1)
    aie.next_bd ^bd
  ^end:
    aie.end
  }
  %c = aie.core(%t) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c6 = arith.constant 6 : i32
    %c23 = arith.constant 23 : i32
    %bz = arith.constant 0.0 : bf16
    %fz = arith.constant 0.0 : f32
    %one = arith.constant 1 : i32
    scf.for %blk = %c0 to %c16 step %c1 {
      aie.use_lock(%lk, AcquireGreaterEqual, %one)
      scf.for %i = %c0 to %c4 step %c1 {
        scf.for %r = %c0 to %c8 step %c1 {
          scf.for %cc = %c0 to %c8 step %c1 {
            %e8 = memref.load %kb[%i, %r, %c0] : memref<4x8x9xi8>
            %c1x = arith.addi %cc, %c1 : index
            %m8 = memref.load %kb[%i, %r, %c1x] : memref<4x8x9xi8>
            %ez = arith.extui %e8 : i8 to i32
            %eb = arith.subi %ez, %c6 : i32
            %es = arith.shli %eb, %c23 : i32
            %sc = arith.bitcast %es : i32 to f32
            %mf = arith.sitofp %m8 : i8 to f32
            %v = arith.mulf %mf, %sc : f32
            %vb = arith.truncf %v : f32 to bf16
            memref.store %vb, %k[%i, %r, %cc] : memref<4x8x8xbf16>
          }
        }
      }
      aie.use_lock(%lkp, Release, %one)
      aie.use_lock(%lk, AcquireGreaterEqual, %one)
      %qv = vector.transfer_read %q[%c0], %bz {in_bounds = [true]} : memref<64xbf16>, vector<64xbf16>
      %qt = vector.shape_cast %qv : vector<64xbf16> to vector<8x8xbf16>
      %qm, %qe = aievec.to_bfp16 %qt : vector<8x8xbf16> -> vector<64xi8>, vector<8xi8>
      scf.for %j = %c0 to %c4 step %c1 {
        %kv = vector.transfer_read %k[%j, %c0, %c0], %bz {in_bounds = [true, true]} : memref<4x8x8xbf16>, vector<8x8xbf16>
        %km, %ke = aievec.to_bfp16 %kv : vector<8x8xbf16> -> vector<64xi8>, vector<8xi8>
        %av = vector.transfer_read %acc[%j, %c0], %fz {in_bounds = [true]} : memref<4x64xf32>, vector<64xf32>
        %a = vector.shape_cast %av : vector<64xf32> to vector<8x8xf32>
        %res = aievec.matmul_aie2p_bfp16 %qm, %qe, %km, %ke, %a : vector<64xi8>, vector<8xi8>, vector<64xi8>, vector<8xi8> into vector<8x8xf32>
        %rv = vector.shape_cast %res : vector<8x8xf32> to vector<64xf32>
        vector.transfer_write %rv, %acc[%j, %c0] {in_bounds = [true]} : vector<64xf32>, memref<4x64xf32>
      }
    }
    aie.end
  }
}

// -----

// No DMA writes the blocks, so they are not known to be complete bfp16
// blocks and the decode stays.
// CHECK-LABEL: aie.device(npu2) @not_from_dma
// CHECK: arith.sitofp
// CHECK-NOT: aievec.load_bfp16
aie.device(npu2) @not_from_dma {
  %t = aie.tile(0, 2)
  %kb = aie.buffer(%t) {sym_name = "kb"} : memref<4x8x9xi8>
  %k = aie.buffer(%t) {sym_name = "k"} : memref<4x8x8xbf16>
  %q = aie.buffer(%t) {sym_name = "q"} : memref<64xbf16>
  %acc = aie.buffer(%t) {sym_name = "acc"} : memref<4x64xf32>
  %lk = aie.lock(%t, 0) {init = 0 : i32}
  %lkp = aie.lock(%t, 1) {init = 1 : i32}
  %c = aie.core(%t) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c6 = arith.constant 6 : i32
    %c23 = arith.constant 23 : i32
    %bz = arith.constant 0.0 : bf16
    %fz = arith.constant 0.0 : f32
    %one = arith.constant 1 : i32
    scf.for %blk = %c0 to %c16 step %c1 {
      aie.use_lock(%lk, AcquireGreaterEqual, %one)
      scf.for %i = %c0 to %c4 step %c1 {
        scf.for %r = %c0 to %c8 step %c1 {
          scf.for %cc = %c0 to %c8 step %c1 {
            %e8 = memref.load %kb[%i, %r, %c0] : memref<4x8x9xi8>
            %c1x = arith.addi %cc, %c1 : index
            %m8 = memref.load %kb[%i, %r, %c1x] : memref<4x8x9xi8>
            %ez = arith.extui %e8 : i8 to i32
            %eb = arith.subi %ez, %c6 : i32
            %es = arith.shli %eb, %c23 : i32
            %sc = arith.bitcast %es : i32 to f32
            %mf = arith.sitofp %m8 : i8 to f32
            %v = arith.mulf %mf, %sc : f32
            %vb = arith.truncf %v : f32 to bf16
            memref.store %vb, %k[%i, %r, %cc] : memref<4x8x8xbf16>
          }
        }
      }
      aie.use_lock(%lkp, Release, %one)
      %qv = vector.transfer_read %q[%c0], %bz {in_bounds = [true]} : memref<64xbf16>, vector<64xbf16>
      %qt = vector.shape_cast %qv : vector<64xbf16> to vector<8x8xbf16>
      %qm, %qe = aievec.to_bfp16 %qt : vector<8x8xbf16> -> vector<64xi8>, vector<8xi8>
      scf.for %j = %c0 to %c4 step %c1 {
        %kv = vector.transfer_read %k[%j, %c0, %c0], %bz {in_bounds = [true, true]} : memref<4x8x8xbf16>, vector<8x8xbf16>
        %km, %ke = aievec.to_bfp16 %kv : vector<8x8xbf16> -> vector<64xi8>, vector<8xi8>
        %av = vector.transfer_read %acc[%j, %c0], %fz {in_bounds = [true]} : memref<4x64xf32>, vector<64xf32>
        %a = vector.shape_cast %av : vector<64xf32> to vector<8x8xf32>
        %res = aievec.matmul_aie2p_bfp16 %qm, %qe, %km, %ke, %a : vector<64xi8>, vector<8xi8>, vector<64xi8>, vector<8xi8> into vector<8x8xf32>
        %rv = vector.shape_cast %res : vector<8x8xf32> to vector<64xf32>
        vector.transfer_write %rv, %acc[%j, %c0] {in_bounds = [true]} : vector<64xf32>, memref<4x64xf32>
      }
    }
    aie.end
  }
}

// -----

// Each tile is the lhs of two bf16 matmuls. Their expansion converts it
// once, so both matmuls use the one read of its block.
// CHECK-LABEL: aie.device(npu2) @two_uses
// CHECK-NOT: sym_name = "k"
// CHECK: aie.core
// CHECK-NOT: arith.sitofp
// CHECK: %{{.*}}, %{{.*}}, %{{.*}}, %[[M:[^,]*]], %[[E:[^ ]*]] = aievec.fifo_ld_pop_bfp16
// CHECK: aievec.matmul_aie2p_bfp16 %[[M]], %[[E]],
// CHECK: aievec.matmul_aie2p_bfp16 %[[M]], %[[E]],
aie.device(npu2) @two_uses {
  %t = aie.tile(0, 2)
  %kb = aie.buffer(%t) {sym_name = "kb"} : memref<4x8x9xi8>
  %k = aie.buffer(%t) {sym_name = "k"} : memref<4x8x8xbf16>
  %q = aie.buffer(%t) {sym_name = "q"} : memref<64xbf16>
  %acc = aie.buffer(%t) {sym_name = "acc"} : memref<4x64xf32>
  %lk = aie.lock(%t, 0) {init = 0 : i32}
  %lkp = aie.lock(%t, 1) {init = 1 : i32}
  %m = aie.mem(%t) {
    %0 = aie.dma_start(S2MM, 0, ^bd, ^end)
  ^bd:
    %m1 = arith.constant 1 : i32
    aie.use_lock(%lkp, AcquireGreaterEqual, %m1)
    aie.dma_bd(%kb : memref<4x8x9xi8> offset = 0 len = 288)
    aie.use_lock(%lk, Release, %m1)
    aie.next_bd ^bd
  ^end:
    aie.end
  }
  %c = aie.core(%t) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c6 = arith.constant 6 : i32
    %c23 = arith.constant 23 : i32
    %bz = arith.constant 0.0 : bf16
    %fz = arith.constant 0.0 : f32
    %one = arith.constant 1 : i32
    scf.for %blk = %c0 to %c16 step %c1 {
      aie.use_lock(%lk, AcquireGreaterEqual, %one)
      scf.for %i = %c0 to %c4 step %c1 {
        scf.for %r = %c0 to %c8 step %c1 {
          scf.for %cc = %c0 to %c8 step %c1 {
            %e8 = memref.load %kb[%i, %r, %c0] : memref<4x8x9xi8>
            %c1x = arith.addi %cc, %c1 : index
            %m8 = memref.load %kb[%i, %r, %c1x] : memref<4x8x9xi8>
            %ez = arith.extui %e8 : i8 to i32
            %eb = arith.subi %ez, %c6 : i32
            %es = arith.shli %eb, %c23 : i32
            %sc = arith.bitcast %es : i32 to f32
            %mf = arith.sitofp %m8 : i8 to f32
            %v = arith.mulf %mf, %sc : f32
            %vb = arith.truncf %v : f32 to bf16
            memref.store %vb, %k[%i, %r, %cc] : memref<4x8x8xbf16>
          }
        }
      }
      aie.use_lock(%lkp, Release, %one)
      %qv = vector.transfer_read %q[%c0], %bz {in_bounds = [true]} : memref<64xbf16>, vector<64xbf16>
      %qt = vector.shape_cast %qv : vector<64xbf16> to vector<8x8xbf16>
      scf.for %j = %c0 to %c4 step %c1 {
        %kv = vector.transfer_read %k[%j, %c0, %c0], %bz {in_bounds = [true, true, true]} : memref<4x8x8xbf16>, vector<1x8x8xbf16>
        %kt = vector.shape_cast %kv : vector<1x8x8xbf16> to vector<8x8xbf16>
        %av = vector.transfer_read %acc[%j, %c0], %fz {in_bounds = [true]} : memref<4x64xf32>, vector<64xf32>
        %a = vector.shape_cast %av : vector<64xf32> to vector<8x8xf32>
        %r0 = aievec.matmul_aie2p %kt, %qt, %a : vector<8x8xbf16>, vector<8x8xbf16> into vector<8x8xf32>
        %r1 = aievec.matmul_aie2p %kt, %qt, %r0 : vector<8x8xbf16>, vector<8x8xbf16> into vector<8x8xf32>
        %rv = vector.shape_cast %r1 : vector<8x8xf32> to vector<64xf32>
        vector.transfer_write %rv, %acc[%j, %c0] {in_bounds = [true]} : vector<64xf32>, memref<4x64xf32>
      }
    }
    aie.end
  }
}

// -----

// The decode loop also counts its rows, and the count is used after it,
// so the loop cannot be removed and the decode stays.
// CHECK-LABEL: aie.device(npu2) @nest_result
// CHECK: arith.sitofp
// CHECK-NOT: aievec.load_bfp16
aie.device(npu2) @nest_result {
  %t = aie.tile(0, 2)
  %kb = aie.buffer(%t) {sym_name = "kb"} : memref<4x8x9xi8>
  %k = aie.buffer(%t) {sym_name = "k"} : memref<4x8x8xbf16>
  %q = aie.buffer(%t) {sym_name = "q"} : memref<64xbf16>
  %acc = aie.buffer(%t) {sym_name = "acc"} : memref<4x64xf32>
  %cnt = aie.buffer(%t) {sym_name = "cnt"} : memref<1xindex>
  %lk = aie.lock(%t, 0) {init = 0 : i32}
  %lkp = aie.lock(%t, 1) {init = 1 : i32}
  %m = aie.mem(%t) {
    %0 = aie.dma_start(S2MM, 0, ^bd, ^end)
  ^bd:
    %m1 = arith.constant 1 : i32
    aie.use_lock(%lkp, AcquireGreaterEqual, %m1)
    aie.dma_bd(%kb : memref<4x8x9xi8> offset = 0 len = 288)
    aie.use_lock(%lk, Release, %m1)
    aie.next_bd ^bd
  ^end:
    aie.end
  }
  %c = aie.core(%t) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c6 = arith.constant 6 : i32
    %c23 = arith.constant 23 : i32
    %bz = arith.constant 0.0 : bf16
    %fz = arith.constant 0.0 : f32
    %one = arith.constant 1 : i32
    scf.for %blk = %c0 to %c16 step %c1 {
      aie.use_lock(%lk, AcquireGreaterEqual, %one)
      %n = scf.for %i = %c0 to %c4 step %c1 iter_args(%a = %c0) -> (index) {
        scf.for %r = %c0 to %c8 step %c1 {
          scf.for %cc = %c0 to %c8 step %c1 {
            %e8 = memref.load %kb[%i, %r, %c0] : memref<4x8x9xi8>
            %c1x = arith.addi %cc, %c1 : index
            %m8 = memref.load %kb[%i, %r, %c1x] : memref<4x8x9xi8>
            %ez = arith.extui %e8 : i8 to i32
            %eb = arith.subi %ez, %c6 : i32
            %es = arith.shli %eb, %c23 : i32
            %sc = arith.bitcast %es : i32 to f32
            %mf = arith.sitofp %m8 : i8 to f32
            %v = arith.mulf %mf, %sc : f32
            %vb = arith.truncf %v : f32 to bf16
            memref.store %vb, %k[%i, %r, %cc] : memref<4x8x8xbf16>
          }
        }
        %a1 = arith.addi %a, %c1 : index
        scf.yield %a1 : index
      }
      memref.store %n, %cnt[%c0] : memref<1xindex>
      aie.use_lock(%lkp, Release, %one)
      %qv = vector.transfer_read %q[%c0], %bz {in_bounds = [true]} : memref<64xbf16>, vector<64xbf16>
      %qt = vector.shape_cast %qv : vector<64xbf16> to vector<8x8xbf16>
      %qm, %qe = aievec.to_bfp16 %qt : vector<8x8xbf16> -> vector<64xi8>, vector<8xi8>
      scf.for %j = %c0 to %c4 step %c1 {
        %kv = vector.transfer_read %k[%j, %c0, %c0], %bz {in_bounds = [true, true]} : memref<4x8x8xbf16>, vector<8x8xbf16>
        %km, %ke = aievec.to_bfp16 %kv : vector<8x8xbf16> -> vector<64xi8>, vector<8xi8>
        %av = vector.transfer_read %acc[%j, %c0], %fz {in_bounds = [true]} : memref<4x64xf32>, vector<64xf32>
        %a = vector.shape_cast %av : vector<64xf32> to vector<8x8xf32>
        %res = aievec.matmul_aie2p_bfp16 %qm, %qe, %km, %ke, %a : vector<64xi8>, vector<8xi8>, vector<64xi8>, vector<8xi8> into vector<8x8xf32>
        %rv = vector.shape_cast %res : vector<8x8xf32> to vector<64xf32>
        vector.transfer_write %rv, %acc[%j, %c0] {in_bounds = [true]} : vector<64xf32>, memref<4x64xf32>
      }
    }
    aie.end
  }
}

// -----

// The core changes a byte of the blocks after decoding them, so the
// decoded tiles differ from the blocks and the decode stays.
// CHECK-LABEL: aie.device(npu2) @bytes_written
// CHECK: arith.sitofp
// CHECK-NOT: aievec.load_bfp16
aie.device(npu2) @bytes_written {
  %t = aie.tile(0, 2)
  %kb = aie.buffer(%t) {sym_name = "kb"} : memref<4x8x9xi8>
  %k = aie.buffer(%t) {sym_name = "k"} : memref<4x8x8xbf16>
  %q = aie.buffer(%t) {sym_name = "q"} : memref<64xbf16>
  %acc = aie.buffer(%t) {sym_name = "acc"} : memref<4x64xf32>
  %lk = aie.lock(%t, 0) {init = 0 : i32}
  %lkp = aie.lock(%t, 1) {init = 1 : i32}
  %m = aie.mem(%t) {
    %0 = aie.dma_start(S2MM, 0, ^bd, ^end)
  ^bd:
    %m1 = arith.constant 1 : i32
    aie.use_lock(%lkp, AcquireGreaterEqual, %m1)
    aie.dma_bd(%kb : memref<4x8x9xi8> offset = 0 len = 288)
    aie.use_lock(%lk, Release, %m1)
    aie.next_bd ^bd
  ^end:
    aie.end
  }
  %c = aie.core(%t) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c6 = arith.constant 6 : i32
    %c23 = arith.constant 23 : i32
    %bz = arith.constant 0.0 : bf16
    %fz = arith.constant 0.0 : f32
    %one = arith.constant 1 : i32
    scf.for %blk = %c0 to %c16 step %c1 {
      aie.use_lock(%lk, AcquireGreaterEqual, %one)
      scf.for %i = %c0 to %c4 step %c1 {
        scf.for %r = %c0 to %c8 step %c1 {
          scf.for %cc = %c0 to %c8 step %c1 {
            %e8 = memref.load %kb[%i, %r, %c0] : memref<4x8x9xi8>
            %c1x = arith.addi %cc, %c1 : index
            %m8 = memref.load %kb[%i, %r, %c1x] : memref<4x8x9xi8>
            %ez = arith.extui %e8 : i8 to i32
            %eb = arith.subi %ez, %c6 : i32
            %es = arith.shli %eb, %c23 : i32
            %sc = arith.bitcast %es : i32 to f32
            %mf = arith.sitofp %m8 : i8 to f32
            %v = arith.mulf %mf, %sc : f32
            %vb = arith.truncf %v : f32 to bf16
            memref.store %vb, %k[%i, %r, %cc] : memref<4x8x8xbf16>
          }
        }
      }
      %z8 = arith.constant 0 : i8
      memref.store %z8, %kb[%c0, %c0, %c1] : memref<4x8x9xi8>
      aie.use_lock(%lkp, Release, %one)
      %qv = vector.transfer_read %q[%c0], %bz {in_bounds = [true]} : memref<64xbf16>, vector<64xbf16>
      %qt = vector.shape_cast %qv : vector<64xbf16> to vector<8x8xbf16>
      %qm, %qe = aievec.to_bfp16 %qt : vector<8x8xbf16> -> vector<64xi8>, vector<8xi8>
      scf.for %j = %c0 to %c4 step %c1 {
        %kv = vector.transfer_read %k[%j, %c0, %c0], %bz {in_bounds = [true, true]} : memref<4x8x8xbf16>, vector<8x8xbf16>
        %km, %ke = aievec.to_bfp16 %kv : vector<8x8xbf16> -> vector<64xi8>, vector<8xi8>
        %av = vector.transfer_read %acc[%j, %c0], %fz {in_bounds = [true]} : memref<4x64xf32>, vector<64xf32>
        %a = vector.shape_cast %av : vector<64xf32> to vector<8x8xf32>
        %res = aievec.matmul_aie2p_bfp16 %qm, %qe, %km, %ke, %a : vector<64xi8>, vector<8xi8>, vector<64xi8>, vector<8xi8> into vector<8x8xf32>
        %rv = vector.shape_cast %res : vector<8x8xf32> to vector<64xf32>
        vector.transfer_write %rv, %acc[%j, %c0] {in_bounds = [true]} : vector<64xf32>, memref<4x64xf32>
      }
    }
    aie.end
  }
}

// -----

// A tile is read before the release of the blocks' lock and converted
// after it. The block is loaded where the tile is read, before the release.
// CHECK-LABEL: aie.device(npu2) @release_first
// CHECK: aie.core
// CHECK-NOT: arith.sitofp
// CHECK: aievec.load_bfp16
// CHECK: aie.use_lock(%{{.*}}, Release
// CHECK-NOT: aievec.load_bfp16
// CHECK: aievec.matmul_aie2p_bfp16
aie.device(npu2) @release_first {
  %t = aie.tile(0, 2)
  %kb = aie.buffer(%t) {sym_name = "kb"} : memref<4x8x9xi8>
  %k = aie.buffer(%t) {sym_name = "k"} : memref<4x8x8xbf16>
  %q = aie.buffer(%t) {sym_name = "q"} : memref<64xbf16>
  %acc = aie.buffer(%t) {sym_name = "acc"} : memref<4x64xf32>
  %lk = aie.lock(%t, 0) {init = 0 : i32}
  %lkp = aie.lock(%t, 1) {init = 1 : i32}
  %m = aie.mem(%t) {
    %0 = aie.dma_start(S2MM, 0, ^bd, ^end)
  ^bd:
    %m1 = arith.constant 1 : i32
    aie.use_lock(%lkp, AcquireGreaterEqual, %m1)
    aie.dma_bd(%kb : memref<4x8x9xi8> offset = 0 len = 288)
    aie.use_lock(%lk, Release, %m1)
    aie.next_bd ^bd
  ^end:
    aie.end
  }
  %c = aie.core(%t) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c6 = arith.constant 6 : i32
    %c23 = arith.constant 23 : i32
    %bz = arith.constant 0.0 : bf16
    %fz = arith.constant 0.0 : f32
    %one = arith.constant 1 : i32
    scf.for %blk = %c0 to %c16 step %c1 {
      aie.use_lock(%lk, AcquireGreaterEqual, %one)
      scf.for %i = %c0 to %c4 step %c1 {
        scf.for %r = %c0 to %c8 step %c1 {
          scf.for %cc = %c0 to %c8 step %c1 {
            %e8 = memref.load %kb[%i, %r, %c0] : memref<4x8x9xi8>
            %c1x = arith.addi %cc, %c1 : index
            %m8 = memref.load %kb[%i, %r, %c1x] : memref<4x8x9xi8>
            %ez = arith.extui %e8 : i8 to i32
            %eb = arith.subi %ez, %c6 : i32
            %es = arith.shli %eb, %c23 : i32
            %sc = arith.bitcast %es : i32 to f32
            %mf = arith.sitofp %m8 : i8 to f32
            %v = arith.mulf %mf, %sc : f32
            %vb = arith.truncf %v : f32 to bf16
            memref.store %vb, %k[%i, %r, %cc] : memref<4x8x8xbf16>
          }
        }
      }
      %k0 = vector.transfer_read %k[%c0, %c0, %c0], %bz {in_bounds = [true, true]} : memref<4x8x8xbf16>, vector<8x8xbf16>
      aie.use_lock(%lkp, Release, %one)
      %k0m, %k0e = aievec.to_bfp16 %k0 : vector<8x8xbf16> -> vector<64xi8>, vector<8xi8>
      %a0v = vector.transfer_read %acc[%c0, %c0], %fz {in_bounds = [true]} : memref<4x64xf32>, vector<64xf32>
      %a0 = vector.shape_cast %a0v : vector<64xf32> to vector<8x8xf32>
      %r0 = aievec.matmul_aie2p_bfp16 %k0m, %k0e, %k0m, %k0e, %a0 : vector<64xi8>, vector<8xi8>, vector<64xi8>, vector<8xi8> into vector<8x8xf32>
      %r0v = vector.shape_cast %r0 : vector<8x8xf32> to vector<64xf32>
      vector.transfer_write %r0v, %acc[%c0, %c0] {in_bounds = [true]} : vector<64xf32>, memref<4x64xf32>
    }
    aie.end
  }
}

// -----

// One tile read feeds conversions in two loops. Both use one load of the
// block.
// CHECK-LABEL: aie.device(npu2) @two_convs
// CHECK-NOT: arith.sitofp
// CHECK: %{{.*}}, %{{.*}}, %{{.*}}, %[[M:[^,]*]], %[[E:[^ ]*]] = aievec.fifo_ld_pop_bfp16
// CHECK: aievec.matmul_aie2p_bfp16 %{{.*}}, %{{.*}}, %[[M]], %[[E]]
// CHECK: aievec.matmul_aie2p_bfp16 %{{.*}}, %{{.*}}, %[[M]], %[[E]]
aie.device(npu2) @two_convs {
  %t = aie.tile(0, 2)
  %kb = aie.buffer(%t) {sym_name = "kb"} : memref<4x8x9xi8>
  %k = aie.buffer(%t) {sym_name = "k"} : memref<4x8x8xbf16>
  %q = aie.buffer(%t) {sym_name = "q"} : memref<64xbf16>
  %acc = aie.buffer(%t) {sym_name = "acc"} : memref<4x64xf32>
  %lk = aie.lock(%t, 0) {init = 0 : i32}
  %lkp = aie.lock(%t, 1) {init = 1 : i32}
  %m = aie.mem(%t) {
    %0 = aie.dma_start(S2MM, 0, ^bd, ^end)
  ^bd:
    %m1 = arith.constant 1 : i32
    aie.use_lock(%lkp, AcquireGreaterEqual, %m1)
    aie.dma_bd(%kb : memref<4x8x9xi8> offset = 0 len = 288)
    aie.use_lock(%lk, Release, %m1)
    aie.next_bd ^bd
  ^end:
    aie.end
  }
  %c = aie.core(%t) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c6 = arith.constant 6 : i32
    %c23 = arith.constant 23 : i32
    %bz = arith.constant 0.0 : bf16
    %fz = arith.constant 0.0 : f32
    %one = arith.constant 1 : i32
    scf.for %blk = %c0 to %c16 step %c1 {
      aie.use_lock(%lk, AcquireGreaterEqual, %one)
      scf.for %i = %c0 to %c4 step %c1 {
        scf.for %r = %c0 to %c8 step %c1 {
          scf.for %cc = %c0 to %c8 step %c1 {
            %e8 = memref.load %kb[%i, %r, %c0] : memref<4x8x9xi8>
            %c1x = arith.addi %cc, %c1 : index
            %m8 = memref.load %kb[%i, %r, %c1x] : memref<4x8x9xi8>
            %ez = arith.extui %e8 : i8 to i32
            %eb = arith.subi %ez, %c6 : i32
            %es = arith.shli %eb, %c23 : i32
            %sc = arith.bitcast %es : i32 to f32
            %mf = arith.sitofp %m8 : i8 to f32
            %v = arith.mulf %mf, %sc : f32
            %vb = arith.truncf %v : f32 to bf16
            memref.store %vb, %k[%i, %r, %cc] : memref<4x8x8xbf16>
          }
        }
      }
      aie.use_lock(%lkp, Release, %one)
      %qv = vector.transfer_read %q[%c0], %bz {in_bounds = [true]} : memref<64xbf16>, vector<64xbf16>
      %qt = vector.shape_cast %qv : vector<64xbf16> to vector<8x8xbf16>
      %qm, %qe = aievec.to_bfp16 %qt : vector<8x8xbf16> -> vector<64xi8>, vector<8xi8>
      scf.for %j = %c0 to %c4 step %c1 {
        %kv = vector.transfer_read %k[%j, %c0, %c0], %bz {in_bounds = [true, true]} : memref<4x8x8xbf16>, vector<8x8xbf16>
        scf.for %u0 = %c0 to %c1 step %c1 {
          %km0, %ke0 = aievec.to_bfp16 %kv : vector<8x8xbf16> -> vector<64xi8>, vector<8xi8>
          %av0 = vector.transfer_read %acc[%j, %c0], %fz {in_bounds = [true]} : memref<4x64xf32>, vector<64xf32>
          %a0 = vector.shape_cast %av0 : vector<64xf32> to vector<8x8xf32>
          %res0 = aievec.matmul_aie2p_bfp16 %qm, %qe, %km0, %ke0, %a0 : vector<64xi8>, vector<8xi8>, vector<64xi8>, vector<8xi8> into vector<8x8xf32>
          %rv0 = vector.shape_cast %res0 : vector<8x8xf32> to vector<64xf32>
          vector.transfer_write %rv0, %acc[%j, %c0] {in_bounds = [true]} : vector<64xf32>, memref<4x64xf32>
        }
        scf.for %u1 = %c0 to %c1 step %c1 {
          %km1, %ke1 = aievec.to_bfp16 %kv : vector<8x8xbf16> -> vector<64xi8>, vector<8xi8>
          %av1 = vector.transfer_read %acc[%j, %c0], %fz {in_bounds = [true]} : memref<4x64xf32>, vector<64xf32>
          %a1 = vector.shape_cast %av1 : vector<64xf32> to vector<8x8xf32>
          %res1 = aievec.matmul_aie2p_bfp16 %qm, %qe, %km1, %ke1, %a1 : vector<64xi8>, vector<8xi8>, vector<64xi8>, vector<8xi8> into vector<8x8xf32>
          %rv1 = vector.shape_cast %res1 : vector<8x8xf32> to vector<64xf32>
          vector.transfer_write %rv1, %acc[%j, %c0] {in_bounds = [true]} : vector<64xf32>, memref<4x64xf32>
        }
      }
    }
    aie.end
  }
}

// -----

// kb's DMA chain acquires the lock that guards kb on the block before kb's
// own. The core's release of that lock moves after the reads of the blocks.
// CHECK-LABEL: aie.device(npu2) @bd_chain
// CHECK: aie.core
// CHECK: aievec.fifo_ld_pop_bfp16
// CHECK: aie.use_lock(%{{.*}}, Release
aie.device(npu2) @bd_chain {
  %t = aie.tile(0, 2)
  %kb = aie.buffer(%t) {sym_name = "kb"} : memref<4x8x9xi8>
  %k = aie.buffer(%t) {sym_name = "k"} : memref<4x8x8xbf16>
  %q = aie.buffer(%t) {sym_name = "q"} : memref<64xbf16>
  %acc = aie.buffer(%t) {sym_name = "acc"} : memref<4x64xf32>
  %lk = aie.lock(%t, 0) {init = 0 : i32}
  %lkp = aie.lock(%t, 1) {init = 1 : i32}
  %other = aie.buffer(%t) {sym_name = "other"} : memref<64xi8>
  %m = aie.mem(%t) {
    %0 = aie.dma_start(S2MM, 0, ^bd0, ^end)
  ^bd0:
    %m1 = arith.constant 1 : i32
    aie.use_lock(%lkp, AcquireGreaterEqual, %m1)
    aie.dma_bd(%other : memref<64xi8> offset = 0 len = 64)
    aie.next_bd ^bd1
  ^bd1:
    %m2 = arith.constant 1 : i32
    aie.dma_bd(%kb : memref<4x8x9xi8> offset = 0 len = 288)
    aie.use_lock(%lk, Release, %m2)
    aie.next_bd ^bd0
  ^end:
    aie.end
  }
  %c = aie.core(%t) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c6 = arith.constant 6 : i32
    %c23 = arith.constant 23 : i32
    %bz = arith.constant 0.0 : bf16
    %fz = arith.constant 0.0 : f32
    %one = arith.constant 1 : i32
    scf.for %blk = %c0 to %c16 step %c1 {
      aie.use_lock(%lk, AcquireGreaterEqual, %one)
      scf.for %i = %c0 to %c4 step %c1 {
        scf.for %r = %c0 to %c8 step %c1 {
          scf.for %cc = %c0 to %c8 step %c1 {
            %e8 = memref.load %kb[%i, %r, %c0] : memref<4x8x9xi8>
            %c1x = arith.addi %cc, %c1 : index
            %m8 = memref.load %kb[%i, %r, %c1x] : memref<4x8x9xi8>
            %ez = arith.extui %e8 : i8 to i32
            %eb = arith.subi %ez, %c6 : i32
            %es = arith.shli %eb, %c23 : i32
            %sc = arith.bitcast %es : i32 to f32
            %mf = arith.sitofp %m8 : i8 to f32
            %v = arith.mulf %mf, %sc : f32
            %vb = arith.truncf %v : f32 to bf16
            memref.store %vb, %k[%i, %r, %cc] : memref<4x8x8xbf16>
          }
        }
      }
      aie.use_lock(%lkp, Release, %one)
      %qv = vector.transfer_read %q[%c0], %bz {in_bounds = [true]} : memref<64xbf16>, vector<64xbf16>
      %qt = vector.shape_cast %qv : vector<64xbf16> to vector<8x8xbf16>
      %qm, %qe = aievec.to_bfp16 %qt : vector<8x8xbf16> -> vector<64xi8>, vector<8xi8>
      scf.for %j = %c0 to %c4 step %c1 {
        %kv = vector.transfer_read %k[%j, %c0, %c0], %bz {in_bounds = [true, true]} : memref<4x8x8xbf16>, vector<8x8xbf16>
        %km, %ke = aievec.to_bfp16 %kv : vector<8x8xbf16> -> vector<64xi8>, vector<8xi8>
        %av = vector.transfer_read %acc[%j, %c0], %fz {in_bounds = [true]} : memref<4x64xf32>, vector<64xf32>
        %a = vector.shape_cast %av : vector<64xf32> to vector<8x8xf32>
        %res = aievec.matmul_aie2p_bfp16 %qm, %qe, %km, %ke, %a : vector<64xi8>, vector<8xi8>, vector<64xi8>, vector<8xi8> into vector<8x8xf32>
        %rv = vector.shape_cast %res : vector<8x8xf32> to vector<64xf32>
        vector.transfer_write %rv, %acc[%j, %c0] {in_bounds = [true]} : vector<64xf32>, memref<4x64xf32>
      }
    }
    aie.end
  }
}

// -----

// kb's DMA chain also moves vb, under locks of its own. Using vb's locks
// between the decode and the reads does not touch kb, so the matmuls still
// read kb's blocks.
// CHECK-LABEL: aie.device(npu2) @shared_chain
// CHECK: aie.core
// CHECK-NOT: arith.sitofp
// CHECK: aievec.fifo_ld_pop_bfp16
aie.device(npu2) @shared_chain {
  %t = aie.tile(0, 2)
  %kb = aie.buffer(%t) {sym_name = "kb"} : memref<4x8x9xi8>
  %k = aie.buffer(%t) {sym_name = "k"} : memref<4x8x8xbf16>
  %q = aie.buffer(%t) {sym_name = "q"} : memref<64xbf16>
  %vbytes = aie.buffer(%t) {sym_name = "vb"} : memref<288xi8>
  %acc = aie.buffer(%t) {sym_name = "acc"} : memref<4x64xf32>
  %lk = aie.lock(%t, 0) {init = 0 : i32}
  %lkp = aie.lock(%t, 1) {init = 1 : i32}
  %lv = aie.lock(%t, 2) {init = 0 : i32}
  %lvp = aie.lock(%t, 3) {init = 1 : i32}
  %m = aie.mem(%t) {
    %0 = aie.dma_start(S2MM, 0, ^bd, ^end)
  ^bd:
    %m1 = arith.constant 1 : i32
    aie.use_lock(%lkp, AcquireGreaterEqual, %m1)
    aie.dma_bd(%kb : memref<4x8x9xi8> offset = 0 len = 288)
    aie.use_lock(%lk, Release, %m1)
    aie.next_bd ^bdv
  ^bdv:
    %m2 = arith.constant 1 : i32
    aie.use_lock(%lvp, AcquireGreaterEqual, %m2)
    aie.dma_bd(%vbytes : memref<288xi8> offset = 0 len = 288)
    aie.use_lock(%lv, Release, %m2)
    aie.next_bd ^bd
  ^end:
    aie.end
  }
  %c = aie.core(%t) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c6 = arith.constant 6 : i32
    %c23 = arith.constant 23 : i32
    %bz = arith.constant 0.0 : bf16
    %fz = arith.constant 0.0 : f32
    %one = arith.constant 1 : i32
    scf.for %blk = %c0 to %c16 step %c1 {
      aie.use_lock(%lk, AcquireGreaterEqual, %one)
      scf.for %i = %c0 to %c4 step %c1 {
        scf.for %r = %c0 to %c8 step %c1 {
          scf.for %cc = %c0 to %c8 step %c1 {
            %e8 = memref.load %kb[%i, %r, %c0] : memref<4x8x9xi8>
            %c1x = arith.addi %cc, %c1 : index
            %m8 = memref.load %kb[%i, %r, %c1x] : memref<4x8x9xi8>
            %ez = arith.extui %e8 : i8 to i32
            %eb = arith.subi %ez, %c6 : i32
            %es = arith.shli %eb, %c23 : i32
            %sc = arith.bitcast %es : i32 to f32
            %mf = arith.sitofp %m8 : i8 to f32
            %v = arith.mulf %mf, %sc : f32
            %vb = arith.truncf %v : f32 to bf16
            memref.store %vb, %k[%i, %r, %cc] : memref<4x8x8xbf16>
          }
        }
      }
      aie.use_lock(%lkp, Release, %one)
      aie.use_lock(%lv, AcquireGreaterEqual, %one)
      aie.use_lock(%lvp, Release, %one)
      %qv = vector.transfer_read %q[%c0], %bz {in_bounds = [true]} : memref<64xbf16>, vector<64xbf16>
      %qt = vector.shape_cast %qv : vector<64xbf16> to vector<8x8xbf16>
      %qm, %qe = aievec.to_bfp16 %qt : vector<8x8xbf16> -> vector<64xi8>, vector<8xi8>
      scf.for %j = %c0 to %c4 step %c1 {
        %kv = vector.transfer_read %k[%j, %c0, %c0], %bz {in_bounds = [true, true]} : memref<4x8x8xbf16>, vector<8x8xbf16>
        %km, %ke = aievec.to_bfp16 %kv : vector<8x8xbf16> -> vector<64xi8>, vector<8xi8>
        %av = vector.transfer_read %acc[%j, %c0], %fz {in_bounds = [true]} : memref<4x64xf32>, vector<64xf32>
        %a = vector.shape_cast %av : vector<64xf32> to vector<8x8xf32>
        %res = aievec.matmul_aie2p_bfp16 %qm, %qe, %km, %ke, %a : vector<64xi8>, vector<8xi8>, vector<64xi8>, vector<8xi8> into vector<8x8xf32>
        %rv = vector.shape_cast %res : vector<8x8xf32> to vector<64xf32>
        vector.transfer_write %rv, %acc[%j, %c0] {in_bounds = [true]} : vector<64xf32>, memref<4x64xf32>
      }
    }
    aie.end
  }
}
