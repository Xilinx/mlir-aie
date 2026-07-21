//===- simple.mlir ---------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: chess
// REQUIRES: peano

// RUN: %aiecc --xchesscc -nv --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host --xclbin-name=aie.xclbin --npu-insts-name=insts.txt %s 2>&1 | FileCheck %s --check-prefix=XCHESSCC
// RUN: %aiecc --no-xchesscc -nv --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host --xclbin-name=aie.xclbin --npu-insts-name=insts.txt %s 2>&1 | FileCheck %s --check-prefix=PEANO

// Note that llc determines the architecture from the llvm IR.
// bootgen runs in-process (no exec line); the xclbin packaging step (xclbinutil)
// is still an external tool.
// XCHESSCC-NOT: {{[^ ]*llc }}
// XCHESSCC: xchesscc_wrapper aie2
// XCHESSCC: xclbinutil
// XCHESSCC-NOT: {{[^ ]*llc }}
// PEANO-NOT: xchesscc_wrapper
// PEANO: {{[^ ]*llc }}
// PEANO-SAME: --march=aie2
// PEANO: xclbinutil

module {
  aie.device(npu1) {
  %12 = aie.tile(1, 2)
  %buf = aie.buffer(%12) : memref<256xi32>
  %4 = aie.core(%12)  {
    %0 = arith.constant 0 : i32
    %1 = arith.constant 0 : index
    memref.store %0, %buf[%1] : memref<256xi32>
    aie.end
  }
  }
}
