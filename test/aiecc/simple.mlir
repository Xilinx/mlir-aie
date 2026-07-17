//===- simple.mlir ---------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: chess
// REQUIRES: peano

// RUN: %PYTHON aiecc.py --aie-generate-core-elfs --xchesscc -nv %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%aie_runtime_lib%/test_lib/include %extraAieCcFlags% -L%aie_runtime_lib%/test_lib/lib -ltest_lib -o test.elf -- %S/test.cpp 2>&1 | FileCheck %s --check-prefix=XCHESSCC
// RUN: %PYTHON aiecc.py --aie-generate-core-elfs --no-xchesscc -nv %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%aie_runtime_lib%/test_lib/include %extraAieCcFlags% -L%aie_runtime_lib%/test_lib/lib -ltest_lib -o test.elf -- %S/test.cpp 2>&1 | FileCheck %s --check-prefix=PEANO
// The NOCOMPILE runs assert that no core compiler is invoked; they request
// input_with_addresses so the driver still produces (compiler-free) output for
// FileCheck to scan instead of an empty graph.
// RUN: %PYTHON aiecc.py --aie-generate-input-with-addresses -nv %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%aie_runtime_lib%/test_lib/include %extraAieCcFlags% -L%aie_runtime_lib%/test_lib/lib -ltest_lib -o test.elf -- %S/test.cpp 2>&1 | FileCheck %s --check-prefix=NOCOMPILE
// RUN: %PYTHON aiecc.py --no-unified --aie-generate-core-elfs --no-xchesscc -nv %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%aie_runtime_lib%/test_lib/include %extraAieCcFlags% -L%aie_runtime_lib%/test_lib/lib -ltest_lib -o test.elf -- %S/test.cpp 2>&1 | FileCheck %s --check-prefix=PEANO
// RUN: %PYTHON aiecc.py --no-unified --aie-generate-input-with-addresses -nv %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%aie_runtime_lib%/test_lib/include %extraAieCcFlags% -L%aie_runtime_lib%/test_lib/lib -ltest_lib -o test.elf -- %S/test.cpp 2>&1 | FileCheck %s --check-prefix=NOCOMPILE

// Note that llc determines the architecture from the llvm IR.

// XCHESSCC-NOT: {{[^ ]*llc }}
// XCHESSCC: xchesscc_wrapper aie
// XCHESSCC-NOT: {{[^ ]*llc }}
// PEANO-NOT: xchesscc_wrapper
// PEANO: {{[^ ]*llc }}
// PEANO-SAME: --march=aie
// PEANO-NOT: xchesscc_wrapper
// NOCOMPILE-NOT: xchesscc_wrapper
// NOCOMPILE-NOT: {{[^ ]*llc }}

module {
  aie.device(xcvc1902) {
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
