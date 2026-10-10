//===- hw_head_of_line_k8.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// HW: PASS for first_a, PASS for first_b (npu2 Strix, 2026-10-10)
// to_a sends 1 + 8 words into a 1-word buffer the core frees only after to_b,
// which shim MM2S 0 sends next. The path buffers 8 words, so first_a
// just finishes; first_b sends to_b first and always finishes.
module {
  aie.device(npu2_1col) {
    %s = aie.tile(0, 0)
    %t = aie.tile(0, 2)
    %bufa = aie.buffer(%t) {sym_name = "bufa"} : memref<1xi32>
    %bufb = aie.buffer(%t) {sym_name = "bufb"} : memref<4xi32>
    %outb = aie.buffer(%t) {sym_name = "outb"} : memref<1xi32>
    %pa = aie.lock(%t, 0) {init = 1 : i32, sym_name = "pa"}
    %ca = aie.lock(%t, 1) {init = 0 : i32, sym_name = "ca"}
    %pb = aie.lock(%t, 2) {init = 1 : i32, sym_name = "pb"}
    %cb = aie.lock(%t, 3) {init = 0 : i32, sym_name = "cb"}
    %po = aie.lock(%t, 4) {init = 1 : i32, sym_name = "po"}
    %co = aie.lock(%t, 5) {init = 0 : i32, sym_name = "co"}
    aie.shim_dma_allocation @to_a(%s, MM2S, 0, <pkt_id = 0>)
    aie.shim_dma_allocation @to_b(%s, MM2S, 0, <pkt_id = 1>)
    aie.shim_dma_allocation @res(%s, S2MM, 0)
    aie.packet_flow(0) { aie.packet_source<%s, DMA : 0> aie.packet_dest<%t, DMA : 0> }
    aie.packet_flow(1) { aie.packet_source<%s, DMA : 0> aie.packet_dest<%t, DMA : 1> }
    aie.flow(%t, DMA : 0, %s, DMA : 0)
    %mem = aie.mem(%t) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^a, ^s1)
    ^a:
      aie.use_lock(%pa, AcquireGreaterEqual, %one)
      aie.dma_bd(%bufa : memref<1xi32> len = 1)
      aie.use_lock(%ca, Release, %one)
      aie.next_bd ^a
    ^s1:
      %1 = aie.dma_start(S2MM, 1, ^b, ^s2)
    ^b:
      aie.use_lock(%pb, AcquireGreaterEqual, %one)
      aie.dma_bd(%bufb : memref<4xi32> len = 4)
      aie.use_lock(%cb, Release, %one)
      aie.next_bd ^b
    ^s2:
      %2 = aie.dma_start(MM2S, 0, ^o, ^end)
    ^o:
      aie.use_lock(%co, AcquireGreaterEqual, %one)
      aie.dma_bd(%outb : memref<1xi32> len = 1)
      aie.use_lock(%po, Release, %one)
      aie.next_bd ^o
    ^end:
      aie.end
    }
    %core = aie.core(%t) {
      %one = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %m = arith.constant 9 : index
      aie.use_lock(%cb, AcquireGreaterEqual, %one)
      %b0 = memref.load %bufb[%c0] : memref<4xi32>
      %b1 = memref.load %bufb[%c1] : memref<4xi32>
      %b2 = memref.load %bufb[%c2] : memref<4xi32>
      %b3 = memref.load %bufb[%c3] : memref<4xi32>
      %s01 = arith.addi %b0, %b1 : i32
      %s23 = arith.addi %b2, %b3 : i32
      %sb = arith.addi %s01, %s23 : i32
      aie.use_lock(%pb, Release, %one)
      %acc = scf.for %i = %c0 to %m step %c1 iter_args(%x = %sb) -> (i32) {
        aie.use_lock(%ca, AcquireGreaterEqual, %one)
        %v = memref.load %bufa[%c0] : memref<1xi32>
        %y = arith.addi %x, %v : i32
        aie.use_lock(%pa, Release, %one)
        scf.yield %y : i32
      }
      aie.use_lock(%po, AcquireGreaterEqual, %one)
      memref.store %acc, %outb[%c0] : memref<1xi32>
      aie.use_lock(%co, Release, %one)
      aie.end
    }
    aie.runtime_sequence @first_a(%a : memref<9xi32>, %b : memref<4xi32>, %r : memref<1xi32>) {
      aiex.npu.dma_memcpy_nd(%r[0, 0, 0, 0][1, 1, 1, 1][0, 0, 0, 1]) {metadata = @res, id = 0 : i64, issue_token = true} : memref<1xi32>
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 9][0, 0, 0, 1]) {metadata = @to_a, id = 1 : i64} : memref<9xi32>
      aiex.npu.dma_memcpy_nd(%b[0, 0, 0, 0][1, 1, 1, 4][0, 0, 0, 1]) {metadata = @to_b, id = 2 : i64} : memref<4xi32>
      aiex.npu.dma_wait {symbol = @res}
    }
    aie.runtime_sequence @first_b(%a : memref<9xi32>, %b : memref<4xi32>, %r : memref<1xi32>) {
      aiex.npu.dma_memcpy_nd(%r[0, 0, 0, 0][1, 1, 1, 1][0, 0, 0, 1]) {metadata = @res, id = 0 : i64, issue_token = true} : memref<1xi32>
      aiex.npu.dma_memcpy_nd(%b[0, 0, 0, 0][1, 1, 1, 4][0, 0, 0, 1]) {metadata = @to_b, id = 2 : i64} : memref<4xi32>
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 9][0, 0, 0, 1]) {metadata = @to_a, id = 1 : i64} : memref<9xi32>
      aiex.npu.dma_wait {symbol = @res}
    }
  }
}
