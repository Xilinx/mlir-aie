//===- bfp16-fifo-aie2p.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --convert-aievec-to-llvm="aie-target=aie2p" | FileCheck %s

// A single block: the FIFO is filled at the block's address, starting from
// an undefined state, and popped once.
// CHECK-LABEL: @load_bfp16
// CHECK: memref.extract_strided_metadata
// CHECK: memref.extract_aligned_pointer_as_index
// CHECK: %[[P:.*]] = llvm.inttoptr
// CHECK: %[[U:.*]] = llvm.mlir.undef : vector<32xi32>
// CHECK: %[[F:.*]] = "xllvm.intr.aie2p.fifo.ld.fill.p0.p0"(%[[P]], %[[U]], %{{.*}})
// CHECK: "xllvm.intr.aie2p.fifo.ld.pop.576.bfp16.p0.p0"
func.func @load_bfp16(%b: memref<288xi8>, %o: index) -> (vector<64xi8>, vector<8xi8>) {
  %m, %e = aievec.load_bfp16 %b[%o] : memref<288xi8> -> vector<64xi8>, vector<8xi8>
  return %m, %e : vector<64xi8>, vector<8xi8>
}

// A stream starts empty. A pop fills the FIFO first unless no_fill is set.
// The address stays a pointer from one access to the next.
// CHECK-LABEL: @load_stream
// CHECK: %[[P:.*]] = llvm.inttoptr
// CHECK-NOT: llvm.ptrtoint
// CHECK: llvm.mlir.undef : vector<32xi32>
// CHECK: "xllvm.intr.aie2p.fifo.ld.fill.p0.p0"(%[[P]],
// CHECK-NEXT: llvm.extractvalue
// CHECK: "xllvm.intr.aie2p.fifo.ld.pop.576.bfp16.p0.p0"
// CHECK-NOT: "xllvm.intr.aie2p.fifo.ld.fill.p0.p0"
// CHECK: "xllvm.intr.aie2p.fifo.ld.pop.576.bfp16.p0.p0"
func.func @load_stream(%b: memref<576xi8>) -> (vector<64xi8>, vector<64xi8>) {
  %c0 = arith.constant 0 : index
  %a, %s, %n = aievec.fifo_start %b[%c0] : memref<576xi8> -> vector<32xi32>
  %a1, %s1, %n1, %m1, %e1 = aievec.fifo_ld_pop_bfp16 %a, %s, %n : vector<32xi32> -> vector<32xi32>, vector<64xi8>, vector<8xi8>
  %a2, %s2, %n2, %m2, %e2 = aievec.fifo_ld_pop_bfp16 %a1, %s1, %n1 no_fill : vector<32xi32> -> vector<32xi32>, vector<64xi8>, vector<8xi8>
  return %m1, %m2 : vector<64xi8>, vector<64xi8>
}

// In a store stream, each block is pushed and then flushed.
// CHECK-LABEL: @store_stream
// CHECK: %[[PU:.*]] = "xllvm.intr.aie2p.fifo.st.push.576.bfp16.p0.p0"
// CHECK: "xllvm.intr.aie2p.fifo.st.flush.p0.p0"
func.func @store_stream(%b: memref<72xi8>, %m: vector<64xi8>, %e: vector<8xi8>) {
  %c0 = arith.constant 0 : index
  %a, %s, %n = aievec.fifo_start %b[%c0] : memref<72xi8> -> vector<32xi32>
  %a1, %s1, %n1 = aievec.fifo_st_push_bfp16 %a, %s, %n, %m, %e : vector<32xi32>, vector<64xi8>, vector<8xi8> -> vector<32xi32>
  return
}

// A loop carries the address as a pointer, not as an integer converted at
// every access.
// CHECK-LABEL: @loop_stream
// CHECK: scf.for {{.*}} -> (!llvm.ptr, vector<32xi32>, i32)
// CHECK-NOT: llvm.ptrtoint
// CHECK-NOT: llvm.inttoptr
// CHECK: "xllvm.intr.aie2p.fifo.ld.pop.576.bfp16.p0.p0"
// CHECK: scf.yield
func.func @loop_stream(%b: memref<576xi8>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %a, %s, %n = aievec.fifo_start %b[%c0] : memref<576xi8> -> vector<32xi32>
  %r:3 = scf.for %i = %c0 to %c8 step %c1 iter_args(%x = %a, %y = %s, %z = %n) -> (!llvm.ptr, vector<32xi32>, i32) {
    %a1, %s1, %n1, %m1, %e1 = aievec.fifo_ld_pop_bfp16 %x, %y, %z : vector<32xi32> -> vector<32xi32>, vector<64xi8>, vector<8xi8>
    scf.yield %a1, %s1, %n1 : !llvm.ptr, vector<32xi32>, i32
  }
  return
}

