//===- loc_sidecar_regdb.mlir -----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Verifies that the locmap sidecar decorates each transaction word with its
// regdb-resolved semantic register name and module. The module names emitted
// must match the JSON keys in aie_registers_aie2.json ("core", "memory",
// "memory_tile", "shim"); a mismatch makes lookupRegisterByOffset silently
// return null and drops the "register"/"register_module" fields entirely.

// RUN: aie-translate --aie-npu-to-binary --aie-npu-emit-locmap --aie-npu-locmap-file=%t.json %s
// RUN: FileCheck %s < %t.json

module {
  aie.device(npu1) {
    aie.runtime_sequence(%arg0: memref<16xf32>) {
      // col 0, row 2 (core tile), tile-relative offset 0x32000 -> Core_Control.
      aiex.npu.write32 { address = 0x232000 : ui32, value = 0x1 : ui32 }
      // col 0, row 0 (shim tile), tile-relative offset 0x14000 -> Lock0_value.
      aiex.npu.write32 { address = 0x14000 : ui32, value = 0x2 : ui32 }
      // col 0, row 1 (mem tile), tile-relative offset 0x91000 -> Performance_Control0.
      aiex.npu.write32 { address = 0x191000 : ui32, value = 0x3 : ui32 }
    }
  }
}

// Core tile register in the "core" module.
// CHECK-DAG: "address": "0x232000"
// CHECK-DAG: "register": "Core_Control"
// CHECK-DAG: "register_module": "core"

// Shim tile register in the "shim" module.
// CHECK-DAG: "address": "0x14000"
// CHECK-DAG: "register": "Lock0_value"
// CHECK-DAG: "register_module": "shim"

// Mem tile register in the "memory_tile" module.
// CHECK-DAG: "address": "0x191000"
// CHECK-DAG: "register": "Performance_Control0"
// CHECK-DAG: "register_module": "memory_tile"
