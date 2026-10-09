//===- only_insts_missing_object.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// An instruction-only build whose sequence names a buffer on a core that links
// an object it cannot find must fail. The probe link measures that object to
// place the buffer, and no real link follows to report it, so a quiet probe
// would leave @rtp placed apart from where a full build places it.

// REQUIRES: peano
// RUN: rm -rf %t.d && mkdir -p %t.d && cd %t.d
// RUN: not %aiecc --get-npu-insts --npu-insts-name=insts.bin %s 2>&1 | FileCheck %s
// RUN: not test -e insts.bin

// CHECK: only_insts_missing_object_kernel.o

module {
  aie.device(npu2) {
    %t02 = aie.tile(0, 2)
    %rtp = aie.buffer(%t02) {sym_name = "rtp"} : memref<4xi32>
    func.func private @kernel(memref<4xi32>) attributes {link_with = "only_insts_missing_object_kernel.o"}
    %core_0_2 = aie.core(%t02) {
      func.call @kernel(%rtp) : (memref<4xi32>) -> ()
      aie.end
    }
    aie.runtime_sequence(%a : memref<64xi32>) {
      %c42 = arith.constant 42 : i32
      aiex.npu.rtp_write(@rtp, 0, %c42) : i32
    }
  }
}
