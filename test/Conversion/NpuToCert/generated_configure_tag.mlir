//===- generated_configure_tag.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt -aie-npu-to-cert -split-input-file %s | FileCheck %s

// The configuration job ({cert.configure}, job 1) comes from the sequence
// tagged aie.generated_configure, not from a user sequence that happens to be
// named `configure`. 0x11111111 = 286331153, 0x22222222 = 572662306,
// 0x33333333 = 858993459, 0x44444444 = 1145324612.

// CHECK-LABEL: aie.device(xcve3858)
// CHECK: aiex.cert.job(2) {
// CHECK-NEXT: aiex.cert.write32(4096, 286331153)
// CHECK-NEXT: }{{$}}
// CHECK-NEXT: aiex.cert.job(1) {
// CHECK-NEXT: aiex.cert.write32(4096, 572662306)
// CHECK-NEXT: } {cert.configure}
module {
  aie.device(xcve3858) @main {
    aie.runtime_sequence @configure() {
      %a = arith.constant 4096 : i32
      %v = arith.constant 0x11111111 : i32
      aiex.npu.write32(%a, %v) : i32, i32
    }
    aie.runtime_sequence @configure0() attributes {aie.generated_configure} {
      %a = arith.constant 4096 : i32
      %v = arith.constant 0x22222222 : i32
      aiex.npu.write32(%a, %v) : i32, i32
    }
  }
}

// -----

// The tagged sequence comes first: once it is converted, the user's
// `configure` is still a user sequence.

// CHECK-LABEL: aie.device(xcve3858)
// CHECK: aiex.cert.job(1) {
// CHECK-NEXT: aiex.cert.write32(4096, 572662306)
// CHECK-NEXT: } {cert.configure}
// CHECK-NEXT: aiex.cert.job(2) {
// CHECK-NEXT: aiex.cert.write32(4096, 286331153)
// CHECK-NEXT: }{{$}}
module {
  aie.device(xcve3858) @main {
    aie.runtime_sequence @configure0() attributes {aie.generated_configure} {
      %a = arith.constant 4096 : i32
      %v = arith.constant 0x22222222 : i32
      aiex.npu.write32(%a, %v) : i32, i32
    }
    aie.runtime_sequence @configure() {
      %a = arith.constant 4096 : i32
      %v = arith.constant 0x11111111 : i32
      aiex.npu.write32(%a, %v) : i32, i32
    }
  }
}

// -----

// Hand-written IR with no tag anywhere: the sequence named `configure` is the
// configuration.

// CHECK-LABEL: aie.device(xcve3858)
// CHECK: aiex.cert.job(1) {
// CHECK-NEXT: aiex.cert.write32(4096, 858993459)
// CHECK-NEXT: } {cert.configure}
// CHECK-NEXT: aiex.cert.job(2) {
// CHECK-NEXT: aiex.cert.write32(4096, 1145324612)
// CHECK-NEXT: }{{$}}
module {
  aie.device(xcve3858) @main {
    aie.runtime_sequence @configure() {
      %a = arith.constant 4096 : i32
      %v = arith.constant 0x33333333 : i32
      aiex.npu.write32(%a, %v) : i32, i32
    }
    aie.runtime_sequence @seq() {
      %a = arith.constant 4096 : i32
      %v = arith.constant 0x44444444 : i32
      aiex.npu.write32(%a, %v) : i32, i32
    }
  }
}
