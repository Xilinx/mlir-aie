// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: not %aiecc --check-lut-banks --xchesscc %s 2>&1 | FileCheck %s
// RUN: not %aiecc --check-lut-banks --xbridge %s 2>&1 | FileCheck %s
// CHECK: --check-lut-banks requires Peano compilation and linking

module {}
