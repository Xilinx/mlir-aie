//===- basic_sequential_fallback_clears_mem_bank.mlir ------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Bank-aware's rollback (deAllocationBuffers) deliberately preserves a
// buffer's `mem_bank` across a failed *bank-aware* attempt, since a later
// bank-aware strategy in the portfolio will still honour it. But when every
// strategy fails and the pass falls back to basic-sequential instead, that
// preserved `mem_bank` is never cleared before basic-sequential's bump
// pointer -- which does not look at mem_bank at all -- hands the buffer a
// completely different address. Left alone, the output would claim "req"
// lives in bank 1 while its real address falls in bank 3: a lie no
// downstream consumer of mem_bank (e.g. DMA routing) could detect on its
// own. `req`'s mem_bank must be dropped once basic-sequential -- which
// cannot honour it -- is the scheme that actually placed it.
//
// f0-f5 exist only to force this specific outcome: sized so every bank-aware
// strategy fails to fit everything (a near-full tile with an awkward mix of
// sizes relative to the 16 kB banks), while linear, bank-oblivious packing
// still fits all of them below the 64 kB tile limit.

// RUN: aie-opt --aie-assign-buffer-addresses %s | FileCheck %s

// CHECK-NOT: mem_bank
// CHECK: %req = aie.buffer(%tile_0_2) {address = {{[0-9]+}} : i32, sym_name = "req"} : memref<128xi8>

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
