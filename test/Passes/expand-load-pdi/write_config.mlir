//===- write_config.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-expand-load-pdi %s | FileCheck %s

// A reload is a reset and a write_config, whose writes the NPU translation
// emits. They are the bytes, and the locmap, of the ops inline-config writes.
// RUN: aie-opt --aie-expand-load-pdi --mlir-print-debuginfo %s -o %t.lazy.mlir
// RUN: aie-opt --aie-expand-load-pdi="inline-config=true" --mlir-print-debuginfo %s -o %t.inline.mlir
// RUN: aie-translate --aie-npu-to-binary --aie-output-binary --aie-device-name=main --aie-sequence-name=first --aie-npu-emit-locmap=%t.lazy.first.json %t.lazy.mlir -o %t.lazy.first.bin
// RUN: aie-translate --aie-npu-to-binary --aie-output-binary --aie-device-name=main --aie-sequence-name=first --aie-npu-emit-locmap=%t.inline.first.json %t.inline.mlir -o %t.inline.first.bin
// RUN: cmp %t.lazy.first.bin %t.inline.first.bin
// RUN: cmp %t.lazy.first.json %t.inline.first.json
// RUN: aie-translate --aie-npu-to-binary --aie-output-binary --aie-device-name=main --aie-sequence-name=second %t.lazy.mlir -o %t.lazy.second.bin
// RUN: aie-translate --aie-npu-to-binary --aie-output-binary --aie-device-name=main --aie-sequence-name=second %t.inline.mlir -o %t.inline.second.bin
// RUN: cmp %t.lazy.second.bin %t.inline.second.bin

// The cert and C++ TXN lowerings write the same ops for a write_config.
// RUN: aie-opt --aie-npu-to-cert %t.lazy.mlir -o %t.lazy.cert.mlir
// RUN: aie-opt --aie-npu-to-cert %t.inline.mlir -o %t.inline.cert.mlir
// RUN: diff %t.lazy.cert.mlir %t.inline.cert.mlir
// RUN: FileCheck %s --check-prefix=CERT < %t.lazy.cert.mlir
// RUN: aie-opt --convert-aiex-to-emitc %t.lazy.mlir -o %t.lazy.cpp.mlir
// RUN: aie-opt --convert-aiex-to-emitc %t.inline.mlir -o %t.inline.cpp.mlir
// RUN: diff %t.lazy.cpp.mlir %t.inline.cpp.mlir

// The transaction conversion leaves each reset device an empty @configure.
// Through page legalization to assembly, every reload still writes its
// configuration.
// RUN: aie-opt --convert-aie-to-transaction %t.lazy.mlir | aie-opt --aie-npu-to-cert | aie-opt --cert-legalize-pages | aie-translate --aie-cert-to-asm -o %t.lazy.asm
// RUN: aie-opt --convert-aie-to-transaction %t.inline.mlir | aie-opt --aie-npu-to-cert | aie-opt --cert-legalize-pages | aie-translate --aie-cert-to-asm -o %t.inline.asm
// RUN: cmp %t.lazy.asm %t.inline.asm
// RUN: FileCheck %s --check-prefix=ASM < %t.lazy.asm

// CERT-NOT: write_config
// CERT: aiex.cert.
// CERT-NOT: write_config

// ASM: LOAD_PDI {{.*}}@empty_0
// ASM: WRITE_32 {{ *}}0x0021f000, 0x00000001
// ASM-NEXT: uC_DMA_WRITE_DES_SYNC {{ *}}@[[C0:chain_[0-9]+]]
// ASM: LOAD_PDI {{.*}}@empty_1
// ASM: WRITE_32 {{ *}}0x0021f000, 0x00000001
// ASM-NEXT: uC_DMA_WRITE_DES_SYNC {{ *}}@[[C1:chain_[0-9]+]]
// ASM: LOAD_PDI {{.*}}@empty_0
// ASM: WRITE_32 {{ *}}0x0021f000, 0x00000001
// ASM-NEXT: uC_DMA_WRITE_DES_SYNC {{ *}}@[[C2:chain_[0-9]+]]
// ASM: LOAD_PDI {{.*}}@empty_1
// ASM: [[C0]]:
// ASM-NEXT: UC_DMA_BD {{ *}}0, 0x00200400, @[[DATA:[a-z_0-9]+]], 4,
// ASM: [[C1]]:
// ASM-NEXT: UC_DMA_BD {{ *}}0, 0x00200400, @[[DATA]], 4,
// ASM: [[C2]]:
// ASM-NEXT: UC_DMA_BD {{ *}}0, 0x00200400, @[[DATA]], 4,
// ASM: [[DATA]]:
// ASM-NEXT: .long {{ *}}0x00000001
// ASM-NEXT: .long {{ *}}0x00000002
// ASM-NEXT: .long {{ *}}0x00000003
// ASM-NEXT: .long {{ *}}0x00000004

// CHECK-LABEL: aie.runtime_sequence @first
// CHECK-NEXT: aiex.npu.load_pdi {device_ref = @empty_0
// CHECK-NEXT: aiex.npu.write_config @init
// CHECK-NEXT: aiex.npu.load_pdi {device_ref = @empty_1
// CHECK-NEXT: aiex.npu.write_config @init
// CHECK-NEXT: {{^}}    }
// CHECK-LABEL: aie.runtime_sequence @second
// CHECK-NEXT: aiex.npu.load_pdi {device_ref = @empty_0
// CHECK-NEXT: aiex.npu.write_config @init
// CHECK-NEXT: aiex.npu.load_pdi {device_ref = @empty_1
// CHECK-NEXT: {{^}}    }

module {
  aie.device(npu2_1col) @init {
    %t = aie.tile(0, 2)
    aie.lock @l(%t, 0) {init = 1 : i32}
    aie.buffer @b(%t) {address = 1024 : i32} : memref<4xi32> = dense<[1, 2, 3, 4]>
  }
  aie.device(npu2_1col) @main {
    %t = aie.tile(0, 2)
    aie.runtime_sequence @first(%arg0: memref<1xi32>) {
      aiex.npu.load_pdi {device_ref = @init}
      aiex.npu.load_pdi {device_ref = @init}
    }
    aie.runtime_sequence @second(%arg0: memref<1xi32>) {
      aiex.npu.load_pdi {device_ref = @init}
    }
  }
}
