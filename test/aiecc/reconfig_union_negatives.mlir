//===- reconfig_union_negatives.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aiecc --get-full-elf --reconfig-method=ctrlpkt %S/Inputs/reconfig_idiomatic_loadpdi.mlir 2>&1 | FileCheck %s --check-prefix=LOADPDI
// RUN: not aiecc --get-full-elf --reconfig-method=ctrlpkt %S/Inputs/reconfig_idiomatic_twoseq.mlir 2>&1 | FileCheck %s --check-prefix=TWOSEQ

// LOADPDI: pre-embedded load_pdi
// TWOSEQ: expected exactly one runtime_sequence
