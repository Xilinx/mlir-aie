//===- missing_chess_wrapper.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// aiecc finds `chess_intrinsic_wrapper.ll` under
// <install prefix>/aie_runtime_lib/<ARCH>/, where the install prefix is the
// parent of the directory holding the aiecc executable. Copying aiecc into a
// private prefix whose aie_runtime_lib is empty reproduces an install built
// without the Chess payload, so no Chess installation or license is needed:
// the diagnostic fires before the linker would be launched. --aietools only
// has to name a directory that exists.

// Chess is Linux-only; the test copies the aiecc executable.
// UNSUPPORTED: system-windows

// RUN: rm -rf missing_chess_wrapper_prefix
// RUN: mkdir -p missing_chess_wrapper_prefix/bin missing_chess_wrapper_prefix/aie_runtime_lib missing_chess_wrapper_prefix/no-aietools
// RUN: cp %aiecc missing_chess_wrapper_prefix/bin/aiecc
// RUN: not ./missing_chess_wrapper_prefix/bin/aiecc %s --xchesscc --get=chesslinked_{0}.ll --aietools=missing_chess_wrapper_prefix/no-aietools --output-dir=missing_chess_wrapper_prefix/out --tmpdir=missing_chess_wrapper_prefix/work -v 2>&1 | FileCheck %s
// RUN: ./missing_chess_wrapper_prefix/bin/aiecc %s --xchesscc --get=chesslinked_{0}.ll --aietools=missing_chess_wrapper_prefix/no-aietools --output-dir=missing_chess_wrapper_prefix/dry --tmpdir=missing_chess_wrapper_prefix/dry-work -v -n 2>&1 | FileCheck %s --check-prefix=DRY

// The diagnostic names the expected path and the build option that produces
// it, and nothing is executed.
// CHECK-NOT: exec:
// CHECK: aiecc: missing Chess intrinsic wrapper: {{.*}}missing_chess_wrapper_prefix/aie_runtime_lib/AIE2P/chess_intrinsic_wrapper.ll
// CHECK: aiecc: install a wheel built with AIETools and AIE2P enabled in AIE_VITIS_COMPONENTS, or rebuild with AIE_BUILD_CHESS_CLANG=ON, AIETools on PATH and AIE2P in AIE_VITIS_COMPONENTS
// CHECK-NOT: exec:

// A dry run never touches the file system, so it still prints the link command.
// DRY-NOT: missing Chess intrinsic wrapper
// DRY: chess-llvm-link

module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    aie.core(%tile) {
      aie.end
    }
  }
}
