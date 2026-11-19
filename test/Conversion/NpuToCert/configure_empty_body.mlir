//===- configure_empty_body.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --convert-aie-to-transaction=device-name=cfg --aie-npu-to-cert --split-input-file %s | FileCheck %s

// An aiex.configure whose body is empty (`{}`) has no block, but still loads
// the device it names. It lowers like any other configure, with nothing to
// run after the load_pdi: the first one builds the section for @cfg from the
// device's generated configuration, the second reuses that section.

// CHECK-LABEL: aie.device(npu2)
// CHECK: aiex.cert.job(2) {
// CHECK-NEXT: aiex.cert.load_pdi({{[0-9]+}}, @cfg)
// CHECK-NEXT: aiex.cert.load_pdi({{[0-9]+}}, @cfg)
// CHECK-NEXT: }
// CHECK: aiex.cert.section @cfg {
// CHECK: aiex.cert.write32(1835008, 3)
// CHECK-NOT: aiex.configure

module {
  aie.device(npu2) @main {
    aie.runtime_sequence @seq() {
      aiex.configure @cfg {
      }
      aiex.configure @cfg {
      }
    }
  }
  aie.device(npu2) @cfg {
    %m = aie.tile(0, 1)
    %l = aie.lock(%m, 0) {init = 3 : i32}
    aie.runtime_sequence @configure() {
    }
  }
}

// -----

// The referenced device's configure sequence is empty too (`{}`), so it has no
// block either. Its section is created with an empty job.

// CHECK-LABEL: aie.device(npu2)
// CHECK: aiex.cert.job(2) {
// CHECK-NEXT: aiex.cert.load_pdi({{[0-9]+}}, @empty)
// CHECK-NEXT: }
// CHECK: aiex.cert.section @empty {
// CHECK-NEXT: aiex.cert.page {
// CHECK-NEXT: aiex.cert.job({{[0-9]+}}) {
// CHECK-NEXT: }
// CHECK-NOT: aiex.configure

module {
  aie.device(npu2) @main {
    aie.runtime_sequence @seq() {
      aiex.configure @empty {
      }
    }
  }
  aie.device(npu2) @empty {
    aie.runtime_sequence @configure() {
    }
  }
}
