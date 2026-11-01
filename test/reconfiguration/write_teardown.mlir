//===- write_teardown.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-expand-load-pdi="teardown=true" %s | FileCheck %s

// The NPU translation, cert and C++ TXN lowerings write the ops inline-config
// writes.
// RUN: aie-opt --aie-expand-load-pdi="teardown=true" --mlir-print-debuginfo %s -o %t.lazy.mlir
// RUN: aie-opt --aie-expand-load-pdi="teardown=true inline-config=true" --mlir-print-debuginfo %s -o %t.inline.mlir
// RUN: aie-translate --aie-npu-to-binary --aie-output-binary --aie-device-name=main --aie-sequence-name=seq --aie-npu-emit-locmap=%t.lazy.json %t.lazy.mlir -o %t.lazy.bin
// RUN: aie-translate --aie-npu-to-binary --aie-output-binary --aie-device-name=main --aie-sequence-name=seq --aie-npu-emit-locmap=%t.inline.json %t.inline.mlir -o %t.inline.bin
// RUN: cmp %t.lazy.bin %t.inline.bin
// RUN: cmp %t.lazy.json %t.inline.json
// RUN: aie-opt --aie-npu-to-cert %t.lazy.mlir -o %t.lazy.cert.mlir
// RUN: aie-opt --aie-npu-to-cert %t.inline.mlir -o %t.inline.cert.mlir
// RUN: diff %t.lazy.cert.mlir %t.inline.cert.mlir
// RUN: aie-opt --convert-aiex-to-emitc %t.lazy.mlir -o %t.lazy.cpp.mlir
// RUN: aie-opt --convert-aiex-to-emitc %t.inline.mlir -o %t.inline.cpp.mlir
// RUN: diff %t.lazy.cpp.mlir %t.inline.cpp.mlir

// CHECK-LABEL: aie.runtime_sequence @seq
// CHECK:      aiex.npu.load_pdi {device_ref = @empty_0
// CHECK-NEXT: aiex.npu.write_config @cfg
// CHECK-NEXT: aiex.npu.write32
// CHECK-NEXT: aiex.npu.write_teardown @cfg
// CHECK-NEXT: aiex.npu.write_config @cfg
// CHECK-NEXT: aiex.npu.write_teardown @cfg
// CHECK-NEXT: {{^}}    }

module {
  aie.device(npu2_1col) @cfg {
    %mem = aie.tile(0, 1)
    %compute = aie.tile(0, 2)
    aie.switchbox(%compute) {
      aie.connect<South : 1, DMA : 0>
    }
    %buf = aie.buffer(%mem) {address = 0 : i32, sym_name = "buf"} : memref<16xi32>
    %memdma = aie.memtile_dma(%mem) {
      %0 = aie.dma_start(MM2S, 1, ^bb1, ^bb2)
    ^bb1:
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.next_bd ^bb1
    ^bb2:
      aie.end
    }
  }

  aie.device(npu2_1col) @main {
    aie.runtime_sequence @seq(%arg0: memref<16xi32>) {
      %c0 = arith.constant 0 : i32
      %a = arith.constant 119316 : i32
      aiex.npu.load_pdi {device_ref = @cfg}
      aiex.npu.write32(%a, %c0) : i32, i32
      aiex.npu.load_pdi {device_ref = @cfg}
    }
  }
}
