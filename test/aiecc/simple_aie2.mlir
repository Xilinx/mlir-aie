//===- simple_aie2.mlir ----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: %PYTHON aiecc.py --basic-alloc-scheme --compile --xchesscc --no-link -nv %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%aie_runtime_lib%/test_lib/include %extraAieCcFlags% -L%aie_runtime_lib%/test_lib/lib -ltest_lib %S/test.cpp -o test.elf | FileCheck %s --check-prefix=XCHESSCC
// RUN: %PYTHON aiecc.py --basic-alloc-scheme --compile --no-xchesscc --no-link -nv %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%aie_runtime_lib%/test_lib/include %extraAieCcFlags% -L%aie_runtime_lib%/test_lib/lib -ltest_lib %S/test.cpp -o test.elf | FileCheck %s --check-prefix=PEANO
// RUN: %PYTHON aiecc.py --basic-alloc-scheme --no-compile --no-link -nv %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%aie_runtime_lib%/test_lib/include %extraAieCcFlags% -L%aie_runtime_lib%/test_lib/lib -ltest_lib %S/test.cpp -o test.elf | FileCheck %s --check-prefix=NOCOMPILE
// RUN: %PYTHON aiecc.py --basic-alloc-scheme --no-unified --compile --no-link --xchesscc -nv %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%aie_runtime_lib%/test_lib/include %extraAieCcFlags% -L%aie_runtime_lib%/test_lib/lib -ltest_lib %S/test.cpp -o test.elf | FileCheck %s --check-prefix=XCHESSCC
// RUN: %PYTHON aiecc.py --basic-alloc-scheme --no-unified --compile --no-link --no-xchesscc -nv %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%aie_runtime_lib%/test_lib/include %extraAieCcFlags% -L%aie_runtime_lib%/test_lib/lib -ltest_lib %S/test.cpp -o test.elf | FileCheck %s --check-prefix=PEANO
// RUN: %PYTHON aiecc.py --basic-alloc-scheme --no-unified --no-compile --no-link -nv %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%aie_runtime_lib%/test_lib/include %extraAieCcFlags% -L%aie_runtime_lib%/test_lib/lib -ltest_lib %S/test.cpp -o test.elf | FileCheck %s --check-prefix=NOCOMPILE

// Note that llc determines the architecture from the llvm IR.
// XCHESSCC-NOT: {{^[^ ]*llc}}
// XCHESSCC: xchesscc_wrapper aie2
// XCHESSCC-NOT: {{^[^ ]*llc}}
// PEANO-NOT: xchesscc_wrapper
// PEANO: {{^[^ ]*llc}}
// PEANO-SAME: --march=aie2
// PEANO-NOT: xchesscc_wrapper
// NOCOMPILE-NOT: xchesscc_wrapper
// NOCOMPILE-NOT: {{^[^ ]*llc}}

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
