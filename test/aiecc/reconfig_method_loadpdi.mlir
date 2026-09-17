//===- reconfig_method_loadpdi.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: peano

// --reconfig-method=loadpdi is the out-of-band, non-persistent delivery variant:
// each config keeps its own npu.load_pdi so the firmware reloads that config's
// whole PDI on every switch. Unlike write32/ctrlpkt it synthesizes NO shared
// @empty standup init (the loadPdiNoInit path). Folds the same two-config build
// the other method tests use (reconfig_twodevice_a.mlir + _b.mlir).
// RUN: rm -rf %t && mkdir -p %t
// RUN: cd %t && aiecc --get-full-elf --reconfig-method=loadpdi --get npu_lowered.mlir --tmpdir=%t %S/Inputs/reconfig_twodevice_a.mlir %S/Inputs/reconfig_twodevice_b.mlir 2>&1
// RUN: cat %t/npu_lowered.mlir | FileCheck %s
// RUN: cat %t/npu_lowered.mlir | FileCheck %s --check-prefix=NOINIT

// Each config folds to its own entry carrying that config's own load_pdi.
// CHECK: aiex.npu.load_pdi {{.*}}@cfg_a
// CHECK: aiex.npu.load_pdi {{.*}}@cfg_b
// CHECK: reconfig_method = "loadpdi"

// No shared @empty reset device is synthesized (that is write32/ctrlpkt's
// main:init standup; loadpdi reloads a full PDI per switch instead).
// NOINIT-NOT: @empty
