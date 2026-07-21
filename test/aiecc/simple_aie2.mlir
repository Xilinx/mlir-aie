//===- simple_aie2.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: chess
// REQUIRES: peano

// RUN: %aiecc --get-core-elfs --xchesscc -nv %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%aie_runtime_lib%/test_lib/include %extraAieCcFlags% -L%aie_runtime_lib%/test_lib/lib -ltest_lib -o test.elf -- %S/test.cpp 2>&1 | FileCheck %s --check-prefix=XCHESSCC
// RUN: %aiecc --get-core-elfs --no-xchesscc -nv %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%aie_runtime_lib%/test_lib/include %extraAieCcFlags% -L%aie_runtime_lib%/test_lib/lib -ltest_lib -o test.elf -- %S/test.cpp 2>&1 | FileCheck %s --check-prefix=PEANO
// The NOCOMPILE runs assert that no core compiler is invoked; they request
// input_with_addresses so the driver still produces (compiler-free) output for
// FileCheck to scan instead of an empty graph.
// RUN: %aiecc --get-input-with-addresses -nv %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%aie_runtime_lib%/test_lib/include %extraAieCcFlags% -L%aie_runtime_lib%/test_lib/lib -ltest_lib -o test.elf -- %S/test.cpp 2>&1 | FileCheck %s --check-prefix=NOCOMPILE
// RUN: %aiecc --no-unified --get-core-elfs --no-xchesscc -nv %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%aie_runtime_lib%/test_lib/include %extraAieCcFlags% -L%aie_runtime_lib%/test_lib/lib -ltest_lib -o test.elf -- %S/test.cpp 2>&1 | FileCheck %s --check-prefix=PEANO
// RUN: %aiecc --no-unified --get-input-with-addresses -nv %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%aie_runtime_lib%/test_lib/include %extraAieCcFlags% -L%aie_runtime_lib%/test_lib/lib -ltest_lib -o test.elf -- %S/test.cpp 2>&1 | FileCheck %s --check-prefix=NOCOMPILE

// Note that llc determines the architecture from the llvm IR.
// XCHESSCC-NOT: {{[^ ]*llc }}
// XCHESSCC: xchesscc_wrapper aie2
// XCHESSCC-NOT: {{[^ ]*llc }}
// PEANO-NOT: xchesscc_wrapper
// PEANO: {{[^ ]*llc }}
// PEANO-SAME: --march=aie2
// PEANO-NOT: xchesscc_wrapper
// NOCOMPILE-NOT: xchesscc_wrapper
// NOCOMPILE-NOT: {{[^ ]*llc }}

module {
  aie.device(xcve2302) {
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
