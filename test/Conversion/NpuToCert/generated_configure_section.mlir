//===- generated_configure_section.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A configured device has its own runtime sequence named `configure`, so
// convert-aie-to-transaction names the generated configuration `configure0`
// and tags it aie.generated_configure. The section built for @cfg must hold the
// generated configuration (memtile lock 0 init = 3, written to 0x1C0000 =
// 1835008), and the user's `configure` (0x5EC0DE01 = 1589698049 written to
// 0x110010 = 1114128) must run where @seq calls it.

// RUN: aie-opt --convert-aie-to-transaction=device-name=cfg %s | FileCheck %s --check-prefix=TAG
// RUN: aie-opt --convert-aie-to-transaction=device-name=cfg --aie-npu-to-cert %s | FileCheck %s

// TAG: aie.runtime_sequence @configure0() attributes {aie.generated_configure}
// TAG: aie.runtime_sequence @configure(%{{.*}}: memref<16xi32>) {

// CHECK-LABEL: aie.device(npu2)
// CHECK: aiex.cert.job(2) {
// CHECK-NEXT: aiex.cert.load_pdi({{[0-9]+}}, @cfg)
// CHECK-NEXT: aiex.cert.write32(1114128, 1589698049)
// CHECK-NEXT: }
// CHECK: aiex.cert.section @cfg {
// CHECK-NOT: 1589698049
// CHECK: aiex.cert.write32(1835008, 3)
// CHECK-NOT: 1589698049
// CHECK-NOT: aie.device

module {
  aie.device(npu2) @main {
    %shim = aie.tile(0, 0)
    aie.runtime_sequence @seq(%a: memref<16xi32>) {
      aiex.configure @cfg {
        aiex.run @configure(%a) : (memref<16xi32>)
      }
    }
  }
  aie.device(npu2) @cfg {
    %m = aie.tile(0, 1)
    %l = aie.lock(%m, 0) {init = 3 : i32}
    aie.runtime_sequence @configure(%a: memref<16xi32>) {
      %addr = arith.constant 0x00110010 : i32
      %val = arith.constant 0x5EC0DE01 : i32
      aiex.npu.write32(%addr, %val) : i32, i32
    }
  }
}
