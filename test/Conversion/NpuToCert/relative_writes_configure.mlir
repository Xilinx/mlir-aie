// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A referenced device's configure sequence is cloned into the main device.
// Its write32, maskwrite32 and maskpoll ops are resolved against the device
// that defines their tiles and buffers, before the clone.

// RUN: aie-opt --aie-npu-to-cert %s | FileCheck %s --implicit-check-not=aiex.npu.write32 --implicit-check-not=aiex.npu.maskwrite32 --implicit-check-not=aiex.npu.maskpoll
// RUN: rm -rf relative_writes_configure && mkdir -p relative_writes_configure
// RUN: aie-opt --aie-npu-to-cert %s -o relative_writes_configure/lowered.mlir 2> relative_writes_configure/stderr.txt
// RUN: FileCheck %s --check-prefix=WARN < relative_writes_configure/stderr.txt
// RUN: aie-translate --aie-cert-to-asm --cert-section-dir=relative_writes_configure relative_writes_configure/lowered.mlir -o relative_writes_configure/main.asm
// RUN: FileCheck %s --check-prefix=ASM < relative_writes_configure/seg.asm

// CHECK-LABEL: aie.device(xcve3858)
// CHECK: aiex.cert.section @seg
// CHECK-NOT: aie.device

// A buffer offset is in words from the buffer's address; tile coordinates
// replace any upper address bits.
// ASM: WRITE_32 0x02200404, 0x00000007
// ASM: WRITE_32 0x02200100, 0x00000008
// ASM: WRITE_32 0x02200100, 0x00000009
// ASM: MASK_WRITE_32 0x02200408, 0x000000ff, 0x00000003
// ASM: MASK_WRITE_32 0x02200104, 0x000000ff, 0x00000004
// ASM: MASK_POLL_32 0x0220040c, 0x00000001, 0x00000001
// ASM: MASK_POLL_32 0x02200108, 0x00000002, 0x00000002

// The mismatch warning is given once, for the op in its own device.
// WARN: warning: address upper bits (0x12300000) don't match row=2 col=1 computed bits (0x2200000)
// WARN-NOT: warning:
module {
  aie.device(xcve3858) @main {
    aie.runtime_sequence @main_seq() {
      aiex.npu.load_pdi {device_ref = @seg}
    }
  }
  aie.device(xcve3858) @seg {
    %tile = aie.tile(1, 2)
    %buffer = aie.buffer(%tile) {address = 1024 : i32, sym_name = "buffer"} : memref<16xi32>
    aie.runtime_sequence @configure() {
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %c3 = arith.constant 3 : i32
      %c4 = arith.constant 4 : i32
      %c7 = arith.constant 7 : i32
      %c8 = arith.constant 8 : i32
      %c9 = arith.constant 9 : i32
      %mask = arith.constant 0xff : i32
      %off0 = arith.constant 0x100 : i32
      %off0_upper = arith.constant 0x12300100 : i32
      %off1 = arith.constant 0x104 : i32
      %off2 = arith.constant 0x108 : i32
      aiex.npu.write32(%c1, %c7) {buffer = @buffer} : i32, i32
      aiex.npu.write32(%off0, %c8) {column = 1 : i32, row = 2 : i32} : i32, i32
      aiex.npu.write32(%off0_upper, %c9) {column = 1 : i32, row = 2 : i32} : i32, i32
      aiex.npu.maskwrite32(%c2, %c3, %mask) {buffer = @buffer} : i32, i32, i32
      aiex.npu.maskwrite32(%off1, %c4, %mask) {column = 1 : i32, row = 2 : i32} : i32, i32, i32
      aiex.npu.maskpoll(%c3, %c1, %c1) {buffer = @buffer} : i32, i32, i32
      aiex.npu.maskpoll(%off2, %c2, %c2) {column = 1 : i32, row = 2 : i32} : i32, i32, i32
    }
  }
}
