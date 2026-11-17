// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A referenced device's configure sequence is cloned into the main device.
// Its blockwrites are resolved against the device that defines their tiles and
// buffers, before the clone.

// RUN: aie-opt --aie-npu-to-cert %s | FileCheck %s --implicit-check-not=aiex.npu.blockwrite
// RUN: rm -rf blockwrite_address_configure && mkdir -p blockwrite_address_configure
// RUN: aie-opt --aie-npu-to-cert %s -o blockwrite_address_configure/lowered.mlir
// RUN: aie-translate --aie-cert-to-asm --cert-section-dir=blockwrite_address_configure blockwrite_address_configure/lowered.mlir -o blockwrite_address_configure/main.asm
// RUN: FileCheck %s --check-prefix=ASM < blockwrite_address_configure/seg.asm

// CHECK-LABEL: aie.device(xcve3858)
// CHECK: aiex.cert.uc_dma_bd @data, 35652612, 4, false
// CHECK: aiex.cert.uc_dma_bd @data, 35651840, 4, false
// CHECK: aiex.cert.section @seg
// CHECK-NOT: aie.device
// ASM: UC_DMA_BD 0, 0x02200404, @data, 4, 0, 0
// ASM: UC_DMA_BD 0, 0x02200100, @data, 4, 0, 0
module {
  aie.device(xcve3858) @main {
    aie.runtime_sequence @main_seq() {
      aiex.npu.load_pdi {device_ref = @seg}
    }
  }
  aie.device(xcve3858) @seg {
    %tile = aie.tile(1, 2)
    %buffer = aie.buffer(%tile) {address = 1024 : i32, sym_name = "buffer"} : memref<16xi32>
    memref.global "private" constant @data : memref<4xi32> = dense<7>
    aie.runtime_sequence @configure() {
      %data = memref.get_global @data : memref<4xi32>
      aiex.npu.blockwrite(%data) {address = 1 : ui32, buffer = @buffer} : memref<4xi32>
      aiex.npu.blockwrite(%data) {address = 0x100 : ui32, column = 1 : i32, row = 2 : i32} : memref<4xi32>
    }
  }
}
