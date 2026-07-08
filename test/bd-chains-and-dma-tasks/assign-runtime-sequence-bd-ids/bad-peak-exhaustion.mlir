//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids --verify-diagnostics --split-input-file %s

// Peak simultaneous BD liveness exceeds the tile's BD pool. An npu2 shim tile
// has 16 BDs; configuring 17 tasks that are all held at once (never freed)
// cannot be allocated. This is a genuine hardware limit and is rejected with a
// clear diagnostic rather than silently overflowing the pool.

aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence(%arg0: memref<8xi16>) {
    %c0_i32 = arith.constant 0 : i32
    %t0  = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      %c8_i32 = arith.constant 8 : i32 aie.dma_bd(%arg0 : memref<8xi16> offset = %c0_i32 len = %c8_i32 sizes = [] strides = []) aie.end }
    %t1  = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      %c8_i32 = arith.constant 8 : i32 aie.dma_bd(%arg0 : memref<8xi16> offset = %c0_i32 len = %c8_i32 sizes = [] strides = []) aie.end }
    %t2  = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      %c8_i32 = arith.constant 8 : i32 aie.dma_bd(%arg0 : memref<8xi16> offset = %c0_i32 len = %c8_i32 sizes = [] strides = []) aie.end }
    %t3  = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      %c8_i32 = arith.constant 8 : i32 aie.dma_bd(%arg0 : memref<8xi16> offset = %c0_i32 len = %c8_i32 sizes = [] strides = []) aie.end }
    %t4  = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      %c8_i32 = arith.constant 8 : i32 aie.dma_bd(%arg0 : memref<8xi16> offset = %c0_i32 len = %c8_i32 sizes = [] strides = []) aie.end }
    %t5  = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      %c8_i32 = arith.constant 8 : i32 aie.dma_bd(%arg0 : memref<8xi16> offset = %c0_i32 len = %c8_i32 sizes = [] strides = []) aie.end }
    %t6  = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      %c8_i32 = arith.constant 8 : i32 aie.dma_bd(%arg0 : memref<8xi16> offset = %c0_i32 len = %c8_i32 sizes = [] strides = []) aie.end }
    %t7  = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      %c8_i32 = arith.constant 8 : i32 aie.dma_bd(%arg0 : memref<8xi16> offset = %c0_i32 len = %c8_i32 sizes = [] strides = []) aie.end }
    %t8  = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      %c8_i32 = arith.constant 8 : i32 aie.dma_bd(%arg0 : memref<8xi16> offset = %c0_i32 len = %c8_i32 sizes = [] strides = []) aie.end }
    %t9  = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      %c8_i32 = arith.constant 8 : i32 aie.dma_bd(%arg0 : memref<8xi16> offset = %c0_i32 len = %c8_i32 sizes = [] strides = []) aie.end }
    %t10 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      %c8_i32 = arith.constant 8 : i32 aie.dma_bd(%arg0 : memref<8xi16> offset = %c0_i32 len = %c8_i32 sizes = [] strides = []) aie.end }
    %t11 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      %c8_i32 = arith.constant 8 : i32 aie.dma_bd(%arg0 : memref<8xi16> offset = %c0_i32 len = %c8_i32 sizes = [] strides = []) aie.end }
    %t12 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      %c8_i32 = arith.constant 8 : i32 aie.dma_bd(%arg0 : memref<8xi16> offset = %c0_i32 len = %c8_i32 sizes = [] strides = []) aie.end }
    %t13 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      %c8_i32 = arith.constant 8 : i32 aie.dma_bd(%arg0 : memref<8xi16> offset = %c0_i32 len = %c8_i32 sizes = [] strides = []) aie.end }
    %t14 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      %c8_i32 = arith.constant 8 : i32 aie.dma_bd(%arg0 : memref<8xi16> offset = %c0_i32 len = %c8_i32 sizes = [] strides = []) aie.end }
    %t15 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      %c8_i32 = arith.constant 8 : i32 aie.dma_bd(%arg0 : memref<8xi16> offset = %c0_i32 len = %c8_i32 sizes = [] strides = []) aie.end }
    // expected-error@+1 {{Too many simultaneously active buffer descriptors}}
    %t16 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      %c8_i32 = arith.constant 8 : i32 aie.dma_bd(%arg0 : memref<8xi16> offset = %c0_i32 len = %c8_i32 sizes = [] strides = []) aie.end }
  }
}
