//===- simple.mlir ---------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aiecc.py --xchesscc -nv --sysroot=%VITIS_SYSROOT% --host-target=aarch64-linux-gnu %s -I%aie_runtime_lib% %aie_runtime_lib%/test_library.cpp %S/test.cpp -o test.elf | FileCheck %s --check-prefix=XCHESSCC
// RUN: aiecc.py --no-xchesscc -nv --sysroot=%VITIS_SYSROOT% --host-target=aarch64-linux-gnu %s -I%aie_runtime_lib% %aie_runtime_lib%/test_library.cpp %S/test.cpp -o test.elf | FileCheck %s --check-prefix=PEANO
// RUN: aiecc.py --no-compile -nv --sysroot=%VITIS_SYSROOT% --host-target=aarch64-linux-gnu %s -I%aie_runtime_lib% %aie_runtime_lib%/test_library.cpp %S/test.cpp -o test.elf | FileCheck %s --check-prefix=NOCOMPILE
// RUN: aiecc.py --no-unified --xchesscc -nv --sysroot=%VITIS_SYSROOT% --host-target=aarch64-linux-gnu %s -I%aie_runtime_lib% %aie_runtime_lib%/test_library.cpp %S/test.cpp -o test.elf | FileCheck %s --check-prefix=XCHESSCC
// RUN: aiecc.py --no-unified --no-xchesscc -nv --sysroot=%VITIS_SYSROOT% --host-target=aarch64-linux-gnu %s -I%aie_runtime_lib% %aie_runtime_lib%/test_library.cpp %S/test.cpp -o test.elf | FileCheck %s --check-prefix=PEANO
// RUN: aiecc.py --no-unified --no-compile -nv --sysroot=%VITIS_SYSROOT% --host-target=aarch64-linux-gnu %s -I%aie_runtime_lib% %aie_runtime_lib%/test_library.cpp %S/test.cpp -o test.elf | FileCheck %s --check-prefix=NOCOMPILE

// XCHESSCC-NOT: {{^llc}}
// XCHESSCC: xchesscc_wrapper
// XCHESSCC-NOT: {{^llc}}
// PEANO-NOT: xchesscc_wrapper
// PEANO: {{^llc}}
// PEANO-SAME: --march=aie
// PEANO-NOT: xchesscc_wrapper
// NOCOMPILE-NOT: xchesscc_wrapper
// NOCOMPILE-NOT: {{^llc}}

module {
  %12 = AIE.tile(1, 2)
  %buf = AIE.buffer(%12) : memref<256xi32>
  %4 = AIE.core(%12)  {
    %0 = arith.constant 0 : i32
    %1 = arith.constant 0 : index
    memref.store %0, %buf[%1] : memref<256xi32>
    AIE.end
  }
}
