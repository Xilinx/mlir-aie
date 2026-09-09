//===- jobs_environment.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// These dry runs exercise option handling without compiling a design.

// RUN: env AIECC_JOBS=1 aiecc --emit-dot > /dev/null
// RUN: env AIECC_JOBS=invalid aiecc -j1 --emit-dot > /dev/null
// RUN: env AIECC_JOBS=invalid aiecc --nthreads=1 --emit-dot > /dev/null

// RUN: not env AIECC_JOBS= aiecc --emit-dot 2>&1 | FileCheck %s
// RUN: not env AIECC_JOBS=-1 aiecc --emit-dot 2>&1 | FileCheck %s
// RUN: not env AIECC_JOBS=' 1' aiecc --emit-dot 2>&1 | FileCheck %s
// RUN: not env AIECC_JOBS=1x aiecc --emit-dot 2>&1 | FileCheck %s
// RUN: not env AIECC_JOBS=4294967296 aiecc --emit-dot 2>&1 | FileCheck %s

// CHECK: aiecc: invalid AIECC_JOBS value '{{.*}}': expected a non-negative integer that fits in an unsigned value
