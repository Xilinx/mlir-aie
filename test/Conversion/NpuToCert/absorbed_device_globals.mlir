// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Preserve payloads and their references across source/destination collisions.
// RUN: aie-opt --aie-npu-to-cert %s | FileCheck %s
// RUN: rm -rf absorbed_device_globals && mkdir -p absorbed_device_globals
// RUN: aie-opt --aie-npu-to-cert %s -o absorbed_device_globals/lowered.mlir
// RUN: aie-translate --aie-cert-to-asm --cert-section-dir=absorbed_device_globals absorbed_device_globals/lowered.mlir -o absorbed_device_globals/main.asm
// RUN: FileCheck %s --check-prefix=ASM < absorbed_device_globals/seg_a.asm

// CHECK: aie.device(xcve3858)
// CHECK-DAG: memref.global "private" constant @payload : memref<2xi32> = dense<[1, 2]>
// CHECK-DAG: memref.global "private" constant @[[A:payload_[0-9]+]] : memref<2xi32> = dense<[3, 4]>
// CHECK-DAG: memref.global "private" constant @payload_0 : memref<2xi32> = dense<[5, 6]>
// CHECK-DAG: memref.global "private" constant @[[B:payload_[0-9]+]] : memref<2xi32> = dense<[7, 8]>
// CHECK-DAG: memref.global "private" constant @[[RESERVED:seg_empty_[0-9]+]] : memref<2xi32> = dense<[11, 12]>
// CHECK-DAG: memref.global "private" constant @unique : memref<2xi32> = dense<[9, 10]>
// CHECK-DAG: uc_dma_bd @[[A]], 4096
// CHECK-DAG: uc_dma_bd @payload_0, 8192
// CHECK-DAG: uc_dma_bd @[[RESERVED]], 20480
// CHECK-DAG: uc_dma_bd @[[B]], 12288
// CHECK-DAG: uc_dma_bd @unique, 16384
// CHECK-DAG: aiex.cert.section @seg_a
// CHECK-DAG: aiex.cert.section @seg_b
// CHECK-DAG: aiex.cert.section @seg_empty
// CHECK-NOT: aie.device
// ASM: uC_DMA_WRITE_DES_SYNC
// ASM: EOF
// ASM: UC_DMA_BD
// ASM-DAG: .long           0x00000003
// ASM-DAG: .long           0x00000004
// ASM-DAG: .long           0x00000005
// ASM-DAG: .long           0x00000006

module {
  aie.device(xcve3858) @main {
    memref.global "private" constant @payload : memref<2xi32> = dense<[1, 2]>
    aie.runtime_sequence @main_seq() {
      aiex.npu.load_pdi {device_ref = @seg_a}
      aiex.npu.load_pdi {device_ref = @seg_b}
      aiex.npu.load_pdi {device_ref = @seg_empty}
    }
  }
  aie.device(xcve3858) @seg_a {
    memref.global "private" constant @payload : memref<2xi32> = dense<[3, 4]>
    memref.global "private" constant @payload_0 : memref<2xi32> = dense<[5, 6]>
    memref.global "private" constant @seg_empty : memref<2xi32> = dense<[11, 12]>
    aie.runtime_sequence @configure() {
      %a = memref.get_global @payload : memref<2xi32>
      aiex.npu.blockwrite(%a) {address = 4096 : ui32} : memref<2xi32>
      %b = memref.get_global @payload_0 : memref<2xi32>
      aiex.npu.blockwrite(%b) {address = 8192 : ui32} : memref<2xi32>
      %reserved = memref.get_global @seg_empty : memref<2xi32>
      aiex.npu.blockwrite(%reserved) {address = 20480 : ui32} : memref<2xi32>
    }
  }
  aie.device(xcve3858) @seg_b {
    memref.global "private" constant @payload : memref<2xi32> = dense<[7, 8]>
    memref.global "private" constant @unique : memref<2xi32> = dense<[9, 10]>
    aie.runtime_sequence @configure() {
      %a = memref.get_global @payload : memref<2xi32>
      aiex.npu.blockwrite(%a) {address = 12288 : ui32} : memref<2xi32>
      %b = memref.get_global @unique : memref<2xi32>
      aiex.npu.blockwrite(%b) {address = 16384 : ui32} : memref<2xi32>
    }
  }
  aie.device(xcve3858) @seg_empty {
    aie.runtime_sequence @configure() {
      %address = arith.constant 0x5000 : i32
      %value = arith.constant 1 : i32
      aiex.npu.write32(%address, %value) : i32, i32
    }
  }
}
