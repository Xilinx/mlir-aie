//===- aie2.mlir -----------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate %S/Inputs/npu1.mlir -aie-generate-target-arch | FileCheck --check-prefix=NPU10 --match-full-lines %s
// NPU10: AIE2

// RUN: aie-translate %S/Inputs/npu1.mlir -aie-generate-target-arch | FileCheck --check-prefix=NPU14 --match-full-lines %s
// NPU14: AIE2

// RUN: aie-translate %S/Inputs/npu2.mlir -aie-generate-target-arch | FileCheck --check-prefix=NPU20 --match-full-lines %s
// NPU20: AIE2p

// RUN: aie-translate %S/Inputs/npu2_4col.mlir -aie-generate-target-arch | FileCheck --check-prefix=NPU24 --match-full-lines %s
// NPU24: AIE2p
