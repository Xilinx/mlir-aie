//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// aiebu block-write-covers-patch invariant on the DYNAMIC BD-pool path, checked
// against the REAL downstream tool. aiebu (the ELF packager) rejects any TXN
// stream where a DDR_PATCH targets an address not covered by a PRECEDING
// block-write; plain XRT dispatch does not enforce this, so the dynamic BD
// lowering could regress silently. The dynamic lowering emits each runtime BD
// as a zero-template block-write over the BD's register block plus a DDR_PATCH
// into that block.
//
// This test lowers a dynamic BD-pool sequence, generates the C++ builder,
// compiles a tiny host program that writes the returned word stream to a raw
// blob, and runs `aiebu-asm -t aie2txn` on it -- exit 0 means aiebu accepted
// the stream. Two shapes are exercised:
//   * straight-line dynamic BD  (Inputs/dma_task_pool_bdid_dyn.mlir)
//   * rolled dynamic scf.for    (this file's @rolled)
//
// NEGATIVE control: a hand-built stream with a patch and no covering
// block-write (the pre-fix shape) is fed to the SAME aiebu and asserted to be
// REJECTED with aiebu's own diagnostic -- so a PASS above cannot be vacuous.
//
// A fast, dependency-free independent parser (Inputs/blockwrite_patch_check.
// cpp) runs first as a smoke check; it agrees with aiebu on both accept and
// reject. aiebu is the authority.
//
//===----------------------------------------------------------------------===//

// REQUIRES: peano, aiebu

// RUN: rm -rf %t.d && mkdir -p %t.d

// ========================================================================
// Shape 1: straight-line dynamic BD pool
// ========================================================================
// RUN: aie-opt --aie-lower-dynamic-bd-pool --canonicalize \
// RUN:   --aie-dma-tasks-to-npu --aie-dma-to-npu \
// RUN:   %S/Inputs/dma_task_pool_bdid_dyn.mlir -o %t.d/straight.mlir
// RUN: aie-translate --aie-npu-to-cpp %t.d/straight.mlir > %t.d/gen_straight.h

// Fast independent smoke check (parser reimplementation of the invariant).
// RUN: %host_clang -std=c++17 -I%S/../../../../include \
// RUN:   -DGEN_HDR='"%t.d/gen_straight.h"' \
// RUN:   -DGEN_FN=generate_txn_main_pool_dynamic -DARGVAL=0 \
// RUN:   %S/Inputs/blockwrite_patch_check.cpp %host_link_flags \
// RUN:   -o %t.d/straight_check.exe
// RUN: %t.d/straight_check.exe | FileCheck %s --check-prefix=SMOKE

// Authoritative check: dump the real word stream and assemble it with aiebu.
// RUN: %host_clang -std=c++17 -I%S/../../../../include \
// RUN:   -DGEN_HDR='"%t.d/gen_straight.h"' \
// RUN:   -DGEN_FN=generate_txn_main_pool_dynamic -DARGVAL=0 \
// RUN:   %S/Inputs/dump_txn_blob.cpp %host_link_flags -o %t.d/dump_straight.exe
// RUN: %t.d/dump_straight.exe %t.d/straight.bin
// RUN: %aiebu_asm -t aie2txn -c %t.d/straight.bin -o %t.d/straight.elf

// SMOKE: self-test negA: correctly rejected
// SMOKE: self-test negB: correctly rejected
// SMOKE: self-test negC: correctly rejected
// SMOKE: self-test posCtl: correctly accepted
// SMOKE: OK: DDR_PATCH
// SMOKE: PASS: block-write-covers-patch invariant holds

// ========================================================================
// Shape 2: rolled dynamic scf.for
// ========================================================================
// RUN: aie-opt --aie-lower-dynamic-bd-pool --canonicalize \
// RUN:   --aie-dma-tasks-to-npu --aie-dma-to-npu %s -o %t.d/rolled.mlir
// RUN: aie-translate --aie-npu-to-cpp %t.d/rolled.mlir > %t.d/gen_rolled.h

// RUN: %host_clang -std=c++17 -I%S/../../../../include \
// RUN:   -DGEN_HDR='"%t.d/gen_rolled.h"' \
// RUN:   -DGEN_FN=generate_txn_main_rolled -DARGVAL=3 \
// RUN:   %S/Inputs/blockwrite_patch_check.cpp %host_link_flags \
// RUN:   -o %t.d/rolled_check.exe
// RUN: %t.d/rolled_check.exe | FileCheck %s --check-prefix=SMOKEROLL

// RUN: %host_clang -std=c++17 -I%S/../../../../include \
// RUN:   -DGEN_HDR='"%t.d/gen_rolled.h"' \
// RUN:   -DGEN_FN=generate_txn_main_rolled -DARGVAL=3 \
// RUN:   %S/Inputs/dump_txn_blob.cpp %host_link_flags -o %t.d/dump_rolled.exe
// RUN: %t.d/dump_rolled.exe %t.d/rolled.bin
// RUN: %aiebu_asm -t aie2txn -c %t.d/rolled.bin -o %t.d/rolled.elf

// SMOKEROLL: self-test negC: correctly rejected
// SMOKEROLL: OK: DDR_PATCH
// SMOKEROLL: PASS: block-write-covers-patch invariant holds

// ========================================================================
// NEGATIVE control: aiebu MUST reject an uncovered-patch stream.
// ========================================================================
// A checker that never fails is worthless; prove the real tool rejects the
// pre-fix shape (patch with no covering block-write). aiebu-asm exits non-zero,
// so we invert with `not` and FileCheck its diagnostic.
// RUN: %host_clang -std=c++17 -I%S/../../../../include -DMAKE_BAD \
// RUN:   %S/Inputs/dump_txn_blob.cpp %host_link_flags -o %t.d/dump_bad.exe
// RUN: %t.d/dump_bad.exe %t.d/bad.bin
// RUN: not %aiebu_asm -t aie2txn -c %t.d/bad.bin -o %t.d/bad.elf 2>&1 \
// RUN:   | FileCheck %s --check-prefix=REJECT

// REJECT: No block-write opcode present before the patch opcode

aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @rolled(%arg0: memref<1024xi32>, %n: index) {
    %c1 = arith.constant 1 : index
    %init = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 1024 sizes = [1, 4, 8, 32] strides = [4096, 512, 32, 1])
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%init)
    %last = scf.for %i = %c1 to %n step %c1 iter_args(%prev = %init) -> (index) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 1024 sizes = [1, 4, 8, 32] strides = [4096, 512, 32, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_free_task(%prev)
      scf.yield %t : index
    }
    aiex.dma_await_task(%last)
    aiex.dma_free_task(%last)
  }
}
