// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-npu-to-cert %s -o %t
// RUN: FileCheck %s < %t
// RUN: aie-translate --aie-cert-to-asm %t | FileCheck %s --check-prefix=ASM

// The poll must remain between the DMA writes: merging their chains across
// the poll would move a write past the queue-space check.
// CHECK: aiex.cert.job(1)
// CHECK: aiex.cert.uc_dma_write_des_sync(@[[BEFORE:chain_[0-9]+]])
// CHECK-NEXT: aiex.cert.maskpoll32(119328, 16777216, 0)
// CHECK-NEXT: aiex.cert.uc_dma_write_des_sync(@[[AFTER:chain_[0-9]+]])
// CHECK-NEXT: aiex.cert.maskpoll32(68276768, 4294967295, 2147483648)
// CHECK-NEXT: aiex.cert.maskpoll32(3147552, 65535, 321)
// CHECK-NOT: aiex.npu.maskpoll

// ASM: START_JOB 1
// ASM: uC_DMA_WRITE_DES_SYNC
// ASM-NEXT: MASK_POLL_32 0x0001d220, 0x01000000, 0x00000000
// ASM-NEXT: uC_DMA_WRITE_DES_SYNC
// ASM-NEXT: MASK_POLL_32 0x0411d220, 0xffffffff, 0x80000000
// ASM-NEXT: MASK_POLL_32 0x00300720, 0x0000ffff, 0x00000141
// ASM-NEXT: END_JOB

aie.device(npu2) {
  %tile = aie.tile(0, 3)
  %buffer = aie.buffer(%tile) {address = 1024 : i32, sym_name = "status"} : memref<128xi32>
  memref.global "private" constant @before : memref<1xi32> = dense<1>
  memref.global "private" constant @after : memref<1xi32> = dense<2>
  aie.runtime_sequence @configure() {
    %address = arith.constant 0x1d220 : i32
    %zero = arith.constant 0 : i32
    %depth = arith.constant 0x1000000 : i32
    %high = arith.constant 0x80000000 : i32
    %all = arith.constant 0xffffffff : i32
    %offset = arith.constant 200 : i32
    %value = arith.constant 321 : i32
    %mask = arith.constant 65535 : i32
    %before = memref.get_global @before : memref<1xi32>
    %after = memref.get_global @after : memref<1xi32>
    aiex.npu.blockwrite(%before) {address = 119316 : ui32} : memref<1xi32>
    aiex.npu.maskpoll(%address, %zero, %depth) : i32, i32, i32
    aiex.npu.blockwrite(%after) {address = 119316 : ui32} : memref<1xi32>
    aiex.npu.maskpoll(%address, %high, %all) {column = 2 : i32, row = 1 : i32} : i32, i32, i32
    aiex.npu.maskpoll(%offset, %value, %mask) {buffer = @status} : i32, i32, i32
  }
}
