//===- bfp16-load-streams.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file --optimize-aievec="aie-target=aie2p" | FileCheck %s

// Each iteration loads the next 8 blocks in order. They become one stream
// started before the loop. Only the first pop of an iteration fills the
// FIFO, since a fill provides enough data for 8 pops. The loop carries the
// stream's address as a pointer.
// CHECK-LABEL: func.func @stream8
// CHECK: aievec.fifo_start %{{.*}}[%c0]
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (!llvm.ptr, vector<32xi32>, i32)
// CHECK: aievec.fifo_ld_pop_bfp16 %{{.*}}, %{{.*}}, %{{.*}} :
// CHECK-COUNT-7: aievec.fifo_ld_pop_bfp16 {{.*}} no_fill
// CHECK-NOT: aievec.load_bfp16
func.func @stream8(%b: memref<2304xi8>) {
  %out = memref.alloca() : memref<64xi8>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c576 = arith.constant 576 : index
  %c72 = arith.constant 72 : index
  %c144 = arith.constant 144 : index
  %c216 = arith.constant 216 : index
  %c288 = arith.constant 288 : index
  %c360 = arith.constant 360 : index
  %c432 = arith.constant 432 : index
  %c504 = arith.constant 504 : index
  scf.for %k = %c0 to %c4 step %c1 {
    %base = arith.muli %k, %c576 : index
    %o0 = arith.addi %base, %c0 : index
    %m0, %e0 = aievec.load_bfp16 %b[%o0] : memref<2304xi8> -> vector<64xi8>, vector<8xi8>
    %o1 = arith.addi %base, %c72 : index
    %m1, %e1 = aievec.load_bfp16 %b[%o1] : memref<2304xi8> -> vector<64xi8>, vector<8xi8>
    %x1 = arith.xori %m0, %m1 : vector<64xi8>
    %o2 = arith.addi %base, %c144 : index
    %m2, %e2 = aievec.load_bfp16 %b[%o2] : memref<2304xi8> -> vector<64xi8>, vector<8xi8>
    %x2 = arith.xori %x1, %m2 : vector<64xi8>
    %o3 = arith.addi %base, %c216 : index
    %m3, %e3 = aievec.load_bfp16 %b[%o3] : memref<2304xi8> -> vector<64xi8>, vector<8xi8>
    %x3 = arith.xori %x2, %m3 : vector<64xi8>
    %o4 = arith.addi %base, %c288 : index
    %m4, %e4 = aievec.load_bfp16 %b[%o4] : memref<2304xi8> -> vector<64xi8>, vector<8xi8>
    %x4 = arith.xori %x3, %m4 : vector<64xi8>
    %o5 = arith.addi %base, %c360 : index
    %m5, %e5 = aievec.load_bfp16 %b[%o5] : memref<2304xi8> -> vector<64xi8>, vector<8xi8>
    %x5 = arith.xori %x4, %m5 : vector<64xi8>
    %o6 = arith.addi %base, %c432 : index
    %m6, %e6 = aievec.load_bfp16 %b[%o6] : memref<2304xi8> -> vector<64xi8>, vector<8xi8>
    %x6 = arith.xori %x5, %m6 : vector<64xi8>
    %o7 = arith.addi %base, %c504 : index
    %m7, %e7 = aievec.load_bfp16 %b[%o7] : memref<2304xi8> -> vector<64xi8>, vector<8xi8>
    %x7 = arith.xori %x6, %m7 : vector<64xi8>
    vector.transfer_write %x7, %out[%c0] {in_bounds = [true]} : vector<64xi8>, memref<64xi8>
  }
  return
}

// -----

// Each iteration loads 2 blocks but advances by 3, so the blocks are not
// consecutive and the loads stay.
// CHECK-LABEL: func.func @gaps
// CHECK-NOT: aievec.fifo_start
// CHECK: aievec.load_bfp16
func.func @gaps(%b: memref<864xi8>) {
  %out = memref.alloca() : memref<64xi8>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c72 = arith.constant 72 : index
  %c216 = arith.constant 216 : index
  scf.for %k = %c0 to %c4 step %c1 {
    %o0 = arith.muli %k, %c216 : index
    %o1 = arith.addi %o0, %c72 : index
    %m0, %e0 = aievec.load_bfp16 %b[%o0] : memref<864xi8> -> vector<64xi8>, vector<8xi8>
    %m1, %e1 = aievec.load_bfp16 %b[%o1] : memref<864xi8> -> vector<64xi8>, vector<8xi8>
    %x = arith.xori %m0, %m1 : vector<64xi8>
    vector.transfer_write %x, %out[%c0] {in_bounds = [true]} : vector<64xi8>, memref<64xi8>
  }
  return
}

// -----

// With 4 blocks per iteration, every pop fills the FIFO first.
// CHECK-LABEL: func.func @stream4
// CHECK: aievec.fifo_start
// CHECK: scf.for
// CHECK-COUNT-4: aievec.fifo_ld_pop_bfp16 %{{.*}}, %{{.*}}, %{{.*}} :
// CHECK-NOT: no_fill
func.func @stream4(%b: memref<1152xi8>) {
  %out = memref.alloca() : memref<64xi8>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c72 = arith.constant 72 : index
  %c144 = arith.constant 144 : index
  %c216 = arith.constant 216 : index
  %c288 = arith.constant 288 : index
  scf.for %k = %c0 to %c4 step %c1 {
    %o0 = arith.muli %k, %c288 : index
    %o1 = arith.addi %o0, %c72 : index
    %o2 = arith.addi %o0, %c144 : index
    %o3 = arith.addi %o0, %c216 : index
    %m0, %e0 = aievec.load_bfp16 %b[%o0] : memref<1152xi8> -> vector<64xi8>, vector<8xi8>
    %m1, %e1 = aievec.load_bfp16 %b[%o1] : memref<1152xi8> -> vector<64xi8>, vector<8xi8>
    %m2, %e2 = aievec.load_bfp16 %b[%o2] : memref<1152xi8> -> vector<64xi8>, vector<8xi8>
    %m3, %e3 = aievec.load_bfp16 %b[%o3] : memref<1152xi8> -> vector<64xi8>, vector<8xi8>
    %x0 = arith.xori %m0, %m1 : vector<64xi8>
    %x1 = arith.xori %m2, %m3 : vector<64xi8>
    %x = arith.xori %x0, %x1 : vector<64xi8>
    vector.transfer_write %x, %out[%c0] {in_bounds = [true]} : vector<64xi8>, memref<64xi8>
  }
  return
}

// -----

// Three buffers are read in order, but a core has only two load FIFOs, so
// no stream is formed.
// CHECK-LABEL: func.func @three_buffers
// CHECK-NOT: aievec.fifo_start
// CHECK-COUNT-3: aievec.load_bfp16
func.func @three_buffers(%a: memref<288xi8>, %b: memref<288xi8>, %c: memref<288xi8>) {
  %out = memref.alloca() : memref<64xi8>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c72 = arith.constant 72 : index
  scf.for %k = %c0 to %c4 step %c1 {
    %o = arith.muli %k, %c72 : index
    %ma, %ea = aievec.load_bfp16 %a[%o] : memref<288xi8> -> vector<64xi8>, vector<8xi8>
    %mb, %eb = aievec.load_bfp16 %b[%o] : memref<288xi8> -> vector<64xi8>, vector<8xi8>
    %mc, %ec = aievec.load_bfp16 %c[%o] : memref<288xi8> -> vector<64xi8>, vector<8xi8>
    %x0 = arith.xori %ma, %mb : vector<64xi8>
    %x = arith.xori %x0, %mc : vector<64xi8>
    vector.transfer_write %x, %out[%c0] {in_bounds = [true]} : vector<64xi8>, memref<64xi8>
  }
  return
}

// -----

// The trip count is not known, so the order of the loads across iterations
// cannot be checked and no stream is formed.
// CHECK-LABEL: func.func @dynamic
// CHECK-NOT: aievec.fifo_start
// CHECK-COUNT-4: aievec.load_bfp16
func.func @dynamic(%b: memref<1152xi8>, %n: index) {
  %out = memref.alloca() : memref<64xi8>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c72 = arith.constant 72 : index
  %c144 = arith.constant 144 : index
  %c216 = arith.constant 216 : index
  %c288 = arith.constant 288 : index
  scf.for %k = %c0 to %n step %c1 {
    %o0 = arith.muli %k, %c288 : index
    %o1 = arith.addi %o0, %c72 : index
    %o2 = arith.addi %o0, %c144 : index
    %o3 = arith.addi %o0, %c216 : index
    %m0, %e0 = aievec.load_bfp16 %b[%o0] : memref<1152xi8> -> vector<64xi8>, vector<8xi8>
    %m1, %e1 = aievec.load_bfp16 %b[%o1] : memref<1152xi8> -> vector<64xi8>, vector<8xi8>
    %m2, %e2 = aievec.load_bfp16 %b[%o2] : memref<1152xi8> -> vector<64xi8>, vector<8xi8>
    %m3, %e3 = aievec.load_bfp16 %b[%o3] : memref<1152xi8> -> vector<64xi8>, vector<8xi8>
    %x0 = arith.xori %m0, %m1 : vector<64xi8>
    %x1 = arith.xori %m2, %m3 : vector<64xi8>
    %x = arith.xori %x0, %x1 : vector<64xi8>
    vector.transfer_write %x, %out[%c0] {in_bounds = [true]} : vector<64xi8>, memref<64xi8>
  }
  return
}

// -----

// Each iteration of the outer loop streams its own row of blocks: the stream
// starts at an offset computed from the outer induction variable, before the
// inner loop.
// CHECK-LABEL: func.func @outer_start
// CHECK: scf.for %[[Z:.*]] = %{{.*}} to %{{.*}} step
// CHECK: %[[S:.*]] = arith.muli %[[Z]], %{{.*}} : index
// CHECK: aievec.fifo_start %{{.*}}[%[[S]]]
// CHECK: scf.for
// CHECK-COUNT-2: aievec.fifo_ld_pop_bfp16
// CHECK-NOT: aievec.load_bfp16
func.func @outer_start(%b: memref<1152xi8>) {
  %out = memref.alloca() : memref<64xi8>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c72 = arith.constant 72 : index
  %c144 = arith.constant 144 : index
  %c576 = arith.constant 576 : index
  scf.for %z = %c0 to %c2 step %c1 {
    %row = arith.muli %z, %c576 : index
    scf.for %k = %c0 to %c4 step %c1 {
      %o = arith.muli %k, %c144 : index
      %o0 = arith.addi %row, %o : index
      %o1 = arith.addi %o0, %c72 : index
      %m0, %e0 = aievec.load_bfp16 %b[%o0] : memref<1152xi8> -> vector<64xi8>, vector<8xi8>
      %m1, %e1 = aievec.load_bfp16 %b[%o1] : memref<1152xi8> -> vector<64xi8>, vector<8xi8>
      %x = arith.xori %m0, %m1 : vector<64xi8>
      vector.transfer_write %x, %out[%c0] {in_bounds = [true]} : vector<64xi8>, memref<64xi8>
    }
  }
  return
}

// -----

// The first block depends on a value that is not the induction variable of a
// loop with static bounds, so the order of the blocks cannot be checked and
// the loads stay.
// CHECK-LABEL: func.func @unknown_start
// CHECK-NOT: aievec.fifo_start
// CHECK-COUNT-2: aievec.load_bfp16
func.func @unknown_start(%b: memref<1152xi8>, %row: index) {
  %out = memref.alloca() : memref<64xi8>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c72 = arith.constant 72 : index
  %c144 = arith.constant 144 : index
  scf.for %k = %c0 to %c4 step %c1 {
    %o = arith.muli %k, %c144 : index
    %o0 = arith.addi %row, %o : index
    %o1 = arith.addi %o0, %c72 : index
    %m0, %e0 = aievec.load_bfp16 %b[%o0] : memref<1152xi8> -> vector<64xi8>, vector<8xi8>
    %m1, %e1 = aievec.load_bfp16 %b[%o1] : memref<1152xi8> -> vector<64xi8>, vector<8xi8>
    %x = arith.xori %m0, %m1 : vector<64xi8>
    vector.transfer_write %x, %out[%c0] {in_bounds = [true]} : vector<64xi8>, memref<64xi8>
  }
  return
}

// -----

// The loop writes the buffer it reads: a pop would read ahead of the write,
// so the loads stay.
// CHECK-LABEL: func.func @written
// CHECK-NOT: aievec.fifo_start
// CHECK-COUNT-2: aievec.load_bfp16
func.func @written(%b: memref<1152xi8>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c72 = arith.constant 72 : index
  %c144 = arith.constant 144 : index
  scf.for %k = %c0 to %c4 step %c1 {
    %o0 = arith.muli %k, %c144 : index
    %o1 = arith.addi %o0, %c72 : index
    %m0, %e0 = aievec.load_bfp16 %b[%o0] : memref<1152xi8> -> vector<64xi8>, vector<8xi8>
    vector.transfer_write %m0, %b[%o1] {in_bounds = [true]} : vector<64xi8>, memref<1152xi8>
    %m1, %e1 = aievec.load_bfp16 %b[%o1] : memref<1152xi8> -> vector<64xi8>, vector<8xi8>
    %x = arith.xori %m0, %m1 : vector<64xi8>
    vector.transfer_write %x, %b[%o0] {in_bounds = [true]} : vector<64xi8>, memref<1152xi8>
  }
  return
}

// -----

// The first block depends on an outer loop with too many trips to check
// each one, so the loads stay (without enumerating the trips).
// CHECK-LABEL: func.func @huge_outer
// CHECK-NOT: aievec.fifo_start
// CHECK-COUNT-2: aievec.load_bfp16
func.func @huge_outer(%b: memref<1152xi8>) {
  %out = memref.alloca() : memref<64xi8>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %big = arith.constant 1099511627776 : index
  %c4 = arith.constant 4 : index
  %c72 = arith.constant 72 : index
  %c144 = arith.constant 144 : index
  %c576 = arith.constant 576 : index
  scf.for %z = %c0 to %big step %c1 {
    %row = arith.muli %z, %c576 : index
    scf.for %k = %c0 to %c4 step %c1 {
      %o = arith.muli %k, %c144 : index
      %o0 = arith.addi %row, %o : index
      %o1 = arith.addi %o0, %c72 : index
      %m0, %e0 = aievec.load_bfp16 %b[%o0] : memref<1152xi8> -> vector<64xi8>, vector<8xi8>
      %m1, %e1 = aievec.load_bfp16 %b[%o1] : memref<1152xi8> -> vector<64xi8>, vector<8xi8>
      %x = arith.xori %m0, %m1 : vector<64xi8>
      vector.transfer_write %x, %out[%c0] {in_bounds = [true]} : vector<64xi8>, memref<64xi8>
    }
  }
  return
}

// -----

// Another core writes kb under locks the loop acquires: a pop could read
// ahead of the write, so the loads stay.
// CHECK-LABEL: aie.device(npu2) @other_core_stream
// CHECK-NOT: aievec.fifo_start
// CHECK: aievec.load_bfp16
aie.device(npu2) @other_core_stream {
  %t = aie.tile(0, 2)
  %t3 = aie.tile(0, 3)
  %kb = aie.buffer(%t) {sym_name = "kb"} : memref<288xi8>
  %l = aie.lock(%t, 2) {init = 0 : i32}
  %lp = aie.lock(%t, 3) {init = 1 : i32}
  %c3 = aie.core(%t3) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c72 = arith.constant 72 : index
    %one = arith.constant 1 : i32
    %v = arith.constant 1 : i8
    scf.for %j = %c0 to %c4 step %c1 {
      aie.use_lock(%lp, AcquireGreaterEqual, %one)
      %o = arith.muli %j, %c72 : index
      memref.store %v, %kb[%o] : memref<288xi8>
      aie.use_lock(%l, Release, %one)
    }
    aie.end
  }
  %c = aie.core(%t) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c72 = arith.constant 72 : index
    %one = arith.constant 1 : i32
    %a0 = arith.constant dense<0.0> : vector<8x8xf32>
    %res = scf.for %j = %c0 to %c4 step %c1 iter_args(%a = %a0) -> vector<8x8xf32> {
      aie.use_lock(%l, AcquireGreaterEqual, %one)
      %ko = arith.muli %j, %c72 : index
      %km, %ke = aievec.load_bfp16 %kb[%ko] : memref<288xi8> -> vector<64xi8>, vector<8xi8>
      aie.use_lock(%lp, Release, %one)
      %r = aievec.matmul_aie2p_bfp16 %km, %ke, %km, %ke, %a : vector<64xi8>, vector<8xi8>, vector<64xi8>, vector<8xi8> into vector<8x8xf32>
      scf.yield %r : vector<8x8xf32>
    }
    aie.end
  }
}

// -----

memref.global "private" @g : memref<1152xi8>

// The loop writes the global through a second handle to it, so the loads
// stay.
// CHECK-LABEL: func.func @same_global
// CHECK-NOT: aievec.fifo_start
// CHECK-COUNT-2: aievec.load_bfp16
func.func @same_global() {
  %b = memref.get_global @g : memref<1152xi8>
  %w = memref.get_global @g : memref<1152xi8>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c72 = arith.constant 72 : index
  %c144 = arith.constant 144 : index
  scf.for %k = %c0 to %c4 step %c1 {
    %o0 = arith.muli %k, %c144 : index
    %o1 = arith.addi %o0, %c72 : index
    %m0, %e0 = aievec.load_bfp16 %b[%o0] : memref<1152xi8> -> vector<64xi8>, vector<8xi8>
    vector.transfer_write %m0, %w[%o1] {in_bounds = [true]} : vector<64xi8>, memref<1152xi8>
    %m1, %e1 = aievec.load_bfp16 %b[%o1] : memref<1152xi8> -> vector<64xi8>, vector<8xi8>
    vector.transfer_write %m1, %w[%o0] {in_bounds = [true]} : vector<64xi8>, memref<1152xi8>
  }
  return
}

// -----

// The loop writes the buffer through the loop-carried handle it passes
// along, so the loads stay.
// CHECK-LABEL: func.func @carried_handle
// CHECK-NOT: aievec.fifo_start
// CHECK-COUNT-2: aievec.load_bfp16
func.func @carried_handle() {
  %b = memref.alloca() : memref<1152xi8>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c72 = arith.constant 72 : index
  %c144 = arith.constant 144 : index
  %r = scf.for %k = %c0 to %c4 step %c1 iter_args(%w = %b) -> (memref<1152xi8>) {
    %o0 = arith.muli %k, %c144 : index
    %o1 = arith.addi %o0, %c72 : index
    %m0, %e0 = aievec.load_bfp16 %b[%o0] : memref<1152xi8> -> vector<64xi8>, vector<8xi8>
    vector.transfer_write %m0, %w[%o1] {in_bounds = [true]} : vector<64xi8>, memref<1152xi8>
    %m1, %e1 = aievec.load_bfp16 %b[%o1] : memref<1152xi8> -> vector<64xi8>, vector<8xi8>
    vector.transfer_write %m1, %w[%o0] {in_bounds = [true]} : vector<64xi8>, memref<1152xi8>
    scf.yield %w : memref<1152xi8>
  }
  return
}
