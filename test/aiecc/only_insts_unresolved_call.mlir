//===- only_insts_unresolved_call.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// An instruction-only build whose kernel calls a function no object defines
// must fail. No real link follows to report it, and a probe that measured the
// stack without the callee's frame would size the core too small.

// REQUIRES: peano
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: clang++ --target=aie2p-none-unknown-elf -std=c++20 -O2 -DNDEBUG -ffunction-sections -fdata-sections -fstack-size-section -c %S/only_insts_unresolved_call_kernel.cc -o %t.d/only_insts_unresolved_call_kernel.o
// RUN: cd %t.d && not %aiecc --get-npu-insts --npu-insts-name=insts.bin %s 2>&1 | FileCheck %s
// RUN: not test -e %t.d/insts.bin

// CHECK: undefined symbol: helper

module {
  aie.device(npu2) {
    %t02 = aie.tile(0, 2)
    %rtp = aie.buffer(%t02) {sym_name = "rtp"} : memref<64xi32>
    func.func private @kernel(memref<64xi32>) attributes {link_with = "only_insts_unresolved_call_kernel.o"}
    %core_0_2 = aie.core(%t02) {
      func.call @kernel(%rtp) : (memref<64xi32>) -> ()
      aie.end
    }
    aie.runtime_sequence(%a : memref<64xi32>) {
      %c42 = arith.constant 42 : i32
      aiex.npu.rtp_write(@rtp, 0, %c42) : i32
    }
  }
}
