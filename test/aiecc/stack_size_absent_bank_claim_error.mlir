//===- stack_size_absent_bank_claim_error.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// stack_size is absent, and the merged kernel's 18 KiB frame is part of the
// core's own object, which was compiled to keep its stack in bank A. Growing
// the stack past bank A would break that claim, so the build fails and names
// the value to declare. Declaring it compiles the core without the claim.

// REQUIRES: peano
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: clang++ --target=aie2-none-unknown-elf -O2 -S -emit-llvm %S/large_stack_kernel.cc -o %t.d/large_stack_kernel.ll
// RUN: cd %t.d && not %aiecc --tmpdir=%t.prj %s 2>&1 | FileCheck %s
// CHECK: error: this core needs a [[N:[0-9]+]]-byte stack. Its own code was compiled to keep its stack frames in memory bank A, but those frames reach {{[0-9]+}} bytes into the stack, past the 16384 bytes left in that bank. Set stack_size = [[N]] (Worker(stack_size=...) in IRON) so that the core is compiled for a stack that spans banks

// RUN: sed 's|} // core|} {stack_size = 20480 : i32}|' %s > %t.d/declared.mlir
// RUN: cd %t.d && %aiecc --tmpdir=%t.declared.prj -v declared.mlir 2>&1 | FileCheck %s --check-prefix=DECLARED --implicit-check-not="aie-stack-addrspace"
// DECLARED: llc {{.*}}--march=aie2

module {
  aie.device(npu1_1col) {
    %tile = aie.tile(0, 2)
    %out = aie.buffer(%tile) {sym_name = "out"} : memref<256xi32>
    func.func private @large_stack(memref<256xi32>) attributes {link_with = "large_stack_kernel.ll", link_with_mode = "merge"}
    aie.core(%tile) {
      func.call @large_stack(%out) : (memref<256xi32>) -> ()
      aie.end
    } // core
  }
}
