//===- generate_pdi.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: chess
// REQUIRES: peano

// Give each run private output/work dirs: Chess drops scratch and aiecc emits
// per-core dirs into the output dir, so concurrent runs sharing a directory (as
// the lit suite does) would clobber each other. The PDIs land in %t; the final
// ls checks both are there.
// RUN: %python aiecc.py -v --xchesscc --xbridge --aie-generate-pdi --pdi-name=MlirAie0.pdi --output-dir=%t --tmpdir=%t.prj %s 2>&1 | FileCheck %s --check-prefix=XCHESSCC
// RUN: %python aiecc.py -v --no-xchesscc --no-xbridge --aie-generate-pdi --pdi-name=MlirAie1.pdi --output-dir=%t --tmpdir=%t.prj %s 2>&1 | FileCheck %s --check-prefix=PEANO

// RUN: ls %t | grep MlirAie | FileCheck %s --check-prefix=CHECK-FILE

// bootgen runs in-process (no exec line); the PDI edge is reported as written.
// XCHESSCC: wrote edge 'MlirAie0.pdi'
// XCHESSCC-NOT: xclbinutil

// PEANO: wrote edge 'MlirAie1.pdi'
// PEANO-NOT: xclbinutil

// CHECK-FILE: MlirAie0.pdi
// CHECK-FILE: MlirAie1.pdi

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
