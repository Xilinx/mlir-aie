//===- backtracking_same_size_buffers.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// One MobileNet bottleneck core under the SA placer: fourteen buffers in three
// sizes fill 64064 of 65536 bytes, and the layout the ranking tries first
// strands the last 2576-byte buffer. Seven buffers share a size, so every order
// they can be placed in leaves the same free runs; the search has to remember
// the states it already exhausted, or it retries each order and runs out of
// budget before it gets back to the choices that matter.

// RUN: aie-opt --aie-assign-buffer-addresses %s | FileCheck %s

// CHECK-COUNT-7: {address = {{[0-9]+}} : i32, mem_bank = {{[0-9]}} : i32, sym_name = "a{{[0-6]}}"}
// CHECK-COUNT-6: {address = {{[0-9]+}} : i32, mem_bank = {{[0-9]}} : i32, sym_name = "b{{[0-5]}}"}
// CHECK: {address = {{[0-9]+}} : i32, mem_bank = {{[0-9]}} : i32, sym_name = "w"}

module @backtracking_same_size_buffers {
  aie.device(npu2) {
    %t = aie.tile(7, 4)
    %a0 = aie.buffer(%t) {sym_name = "a0"} : memref<2576xui8>
    %a1 = aie.buffer(%t) {sym_name = "a1"} : memref<2576xui8>
    %a2 = aie.buffer(%t) {sym_name = "a2"} : memref<2576xui8>
    %a3 = aie.buffer(%t) {sym_name = "a3"} : memref<2576xui8>
    %a4 = aie.buffer(%t) {sym_name = "a4"} : memref<2576xui8>
    %a5 = aie.buffer(%t) {sym_name = "a5"} : memref<2576xui8>
    %a6 = aie.buffer(%t) {sym_name = "a6"} : memref<2576xui8>
    %b0 = aie.buffer(%t) {sym_name = "b0"} : memref<6720xui8>
    %b1 = aie.buffer(%t) {sym_name = "b1"} : memref<6720xui8>
    %b2 = aie.buffer(%t) {sym_name = "b2"} : memref<6720xui8>
    %b3 = aie.buffer(%t) {sym_name = "b3"} : memref<6720xui8>
    %b4 = aie.buffer(%t) {sym_name = "b4"} : memref<6720xui8>
    %b5 = aie.buffer(%t) {sym_name = "b5"} : memref<6720xui8>
    %w = aie.buffer(%t) {sym_name = "w"} : memref<4320xi8>
    aie.core(%t) {
      aie.end
    }
  }
}
