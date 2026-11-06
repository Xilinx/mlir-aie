// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-objectfifo-lower-cores --aie-objectFifo-unroll %s | FileCheck %s --implicit-check-not='memref.alloca() : memref<index>' --implicit-check-not=scf.index_switch

// Both branches send buffer 0; the release after their merge sends buffer 1.
// CHECK: scf.if
// CHECK: aie.dma_push_queue(MM2S : 0) {bd_tag = 7 : i32}
// CHECK: } else {
// CHECK: aie.dma_push_queue(MM2S : 0) {bd_tag = 7 : i32}
// CHECK: aie.dma_push_queue(MM2S : 0) {bd_tag = 8 : i32}

module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %b0 = aie.buffer(%tile) {sym_name = "b0"} : memref<16xi32>
    %b1 = aie.buffer(%tile) {sym_name = "b1"} : memref<16xi32>
    %condition = aie.buffer(%tile) {sym_name = "condition"} : memref<1xi32>
    %free = aie.lock(%tile) {sym_name = "free", init = 2 : i32}
    %full = aie.lock(%tile) {sym_name = "full", init = 0 : i32}
    aie.objectfifo.pool @p(%tile) {depth = 2 : i32, buffers = [@b0, @b1]} : memref<16xi32> {
      aie.objectfifo.segment @all {offset = 0 : i32, size = 16 : i32, produceLock = @free, consumeLock = @full}
    }
    aie.objectfifo.core_endpoint @fill(%tile) fills @p {core_push = #aie.core_push<channel = 0, bd_tags = [7, 8]>}
    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %zero = arith.constant 0 : i32
      %value = memref.load %condition[%c0] : memref<1xi32>
      %cond = arith.cmpi ne, %value, %zero : i32
      scf.if %cond {
        %a = aie.objectfifo.acquire @fill(1) : memref<16xi32>
        aie.objectfifo.release @fill(1)
      } else {
        %b = aie.objectfifo.acquire @fill(1) : memref<16xi32>
        aie.objectfifo.release @fill(1)
      }
      %c = aie.objectfifo.acquire @fill(1) : memref<16xi32>
      aie.objectfifo.release @fill(1)
      aie.end
    }
  }
}
