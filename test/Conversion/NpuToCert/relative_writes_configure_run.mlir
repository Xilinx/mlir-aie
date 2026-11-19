// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Relative write32, maskwrite32 and maskpoll ops reached through aiex.configure
// are resolved against the device that defines their tiles and buffers: those
// in its configure sequence, cloned into the section, and those in a sequence
// inlined by aiex.run.

// RUN: aie-opt --aie-npu-to-cert %s | FileCheck %s --implicit-check-not=aiex.npu.write32 --implicit-check-not=aiex.npu.maskwrite32 --implicit-check-not=aiex.npu.maskpoll

// The buffer is at 1024 in tile (1, 2), whose upper address bits are 0x2200000.
// CHECK-LABEL: aie.device(xcve3858)
// CHECK: aiex.cert.job(2) {
// CHECK-NEXT: aiex.cert.load_pdi({{[0-9]+}}, @seg)
// 0x2200408
// CHECK-NEXT: aiex.cert.maskpoll32(35652616, 8, 8)
// CHECK-NEXT: aiex.cert.write32(35652616, 8)
// 0x2200108
// CHECK-NEXT: aiex.cert.maskwrite32(35651848, 255, 8)
// CHECK-NEXT: }
// CHECK: aiex.cert.section @seg {
// 0x2200404
// CHECK: aiex.cert.write32(35652612, 7)
// 0x2200104
// CHECK-NEXT: aiex.cert.maskwrite32(35651844, 255, 7)
// CHECK-NOT: aie.device
module {
  aie.device(xcve3858) @main {
    aie.runtime_sequence @seq() {
      aiex.configure @seg {
        %w = arith.constant 0 : i32
        aiex.run @work(%w) : (i32)
      }
    }
  }
  aie.device(xcve3858) @seg {
    %tile = aie.tile(1, 2)
    %buffer = aie.buffer(%tile) {address = 1024 : i32, sym_name = "buffer"} : memref<16xi32>
    aie.runtime_sequence @configure() {
      %c1 = arith.constant 1 : i32
      %c7 = arith.constant 7 : i32
      %mask = arith.constant 0xff : i32
      %off = arith.constant 0x104 : i32
      aiex.npu.write32(%c1, %c7) {buffer = @buffer} : i32, i32
      aiex.npu.maskwrite32(%off, %c7, %mask) {column = 1 : i32, row = 2 : i32} : i32, i32, i32
    }
    aie.runtime_sequence @work(%unused: i32) {
      %c2 = arith.constant 2 : i32
      %c8 = arith.constant 8 : i32
      %mask = arith.constant 0xff : i32
      %off = arith.constant 0x108 : i32
      aiex.npu.maskpoll(%c2, %c8, %c8) {buffer = @buffer} : i32, i32, i32
      aiex.npu.write32(%c2, %c8) {buffer = @buffer} : i32, i32
      aiex.npu.maskwrite32(%off, %c8, %mask) {column = 1 : i32, row = 2 : i32} : i32, i32, i32
    }
  }
}
