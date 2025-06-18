//===- aie2.mlir -----------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

// RUN: echo 'aie.device(npu1){aie.end}' | aie-translate -aie-generate-target-arch | FileCheck --check-prefix=NPU10 --match-full-lines %s
// NPU10: AIE2

// RUN: echo 'aie.device(npu1){aie.end}' | aie-translate -aie-generate-target-arch | FileCheck --check-prefix=NPU14 --match-full-lines %s
// NPU14: AIE2

// RUN: echo 'aie.device(npu2){aie.end}' | aie-translate -aie-generate-target-arch | FileCheck --check-prefix=NPU20 --match-full-lines %s
// NPU20: AIE2p

// RUN: echo 'aie.device(npu2_4col){aie.end}' | aie-translate -aie-generate-target-arch | FileCheck --check-prefix=NPU24 --match-full-lines %s
// NPU24: AIE2p
