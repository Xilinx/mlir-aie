//===- signature_bank_constraint.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A kernel that qualifies a pointer parameter with `__aie_dm_resource_a` is
// telling Peano's bank-conflict model which bank that pointer reaches. The
// qualifier is an address space, and 5..8 are banks a..d, so it reaches MLIR as
// a memory space. Placement honors it, which is what makes the assumption Peano
// schedules on actually true.
//
// `func.call` requires operand types to match the callee's signature, so the
// buffer and the kernel declaration are kept in agreement by the verifier and
// the buffer's own type is the single thing placement has to read.

// RUN: aie-opt --aie-assign-buffer-addresses %s | FileCheck %s

// CHECK: mem_bank = 1 : i32, sym_name = "in_bank_b"
// CHECK: mem_bank = 3 : i32, sym_name = "in_bank_d"
// An unqualified buffer keeps whatever bank placement finds for it.
// CHECK: sym_name = "unconstrained"

module {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %in_bank_b = aie.buffer(%tile_0_2) {sym_name = "in_bank_b"} : memref<64xi8, 6>
    %in_bank_d = aie.buffer(%tile_0_2) {sym_name = "in_bank_d"} : memref<64xi8, 8>
    %unconstrained = aie.buffer(%tile_0_2) {sym_name = "unconstrained"} : memref<64xi8>

    // Parameters 0 and 1 name banks b and d; the third names none.
    func.func private @gather(memref<64xi8, 6>, memref<64xi8, 8>, memref<64xi8>)

    %core_0_2 = aie.core(%tile_0_2) {
      func.call @gather(%in_bank_b, %in_bank_d, %unconstrained)
          : (memref<64xi8, 6>, memref<64xi8, 8>, memref<64xi8>) -> ()
      aie.end
    } { stack_size = 1024 : i32 }
  }
}

