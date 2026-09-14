//===- bank_aware_oom_refuses_to_drop_mem_bank.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// "req"'s own bank pin is satisfiable. Bank-aware allocation fails here only
// because f0-f5 do not all fit at once; their sizes force that outcome, while
// still fitting under linear, bank-oblivious packing. Basic-sequential has no
// notion of banks and would place "req" wherever its bump pointer landed, which
// no downstream consumer of mem_bank (DMA routing, for instance) can detect.
// mem_bank is a hard constraint, so the auto scheme reports a terminal error
// here instead of retrying basic-sequential.

// RUN: not aie-opt --aie-assign-buffer-addresses %s 2>&1 | FileCheck %s

// CHECK: error: 'aie.tile' op bank-aware allocation failed; falling back to basic-sequential would silently drop the mem_bank pin on: "req"

module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %req = aie.buffer(%t) {sym_name = "req", mem_bank = 1 : i32} : memref<128xi8>
    %f0 = aie.buffer(%t) {sym_name = "f0"} : memref<13952xi8>
    %f1 = aie.buffer(%t) {sym_name = "f1"} : memref<11744xi8>
    %f2 = aie.buffer(%t) {sym_name = "f2"} : memref<10688xi8>
    %f3 = aie.buffer(%t) {sym_name = "f3"} : memref<12352xi8>
    %f4 = aie.buffer(%t) {sym_name = "f4"} : memref<11648xi8>
    %f5 = aie.buffer(%t) {sym_name = "f5"} : memref<4000xi8>
    aie.core(%t) { aie.end } {stack_size = 1024 : i32}
  }
}
