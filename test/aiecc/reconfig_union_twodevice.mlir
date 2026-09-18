//===- reconfig_union_twodevice.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The fold writes the merged module to <tmpdir>/config_union.mlir before any
// build, so we FileCheck that artifact (aiecc's exit is ignored -- the union is
// on disk regardless of whether the later build stages need peano).
// RUN: rm -rf %t && mkdir -p %t
// RUN: cd %t && aiecc --get-full-elf --reconfig-method=ctrlpkt --get npu_lowered.mlir --tmpdir=%t %S/Inputs/reconfig_twodevice_a.mlir %S/Inputs/reconfig_twodevice_b.mlir > /dev/null 2>&1 ; cat %t/config_union.mlir | FileCheck %s

// The two-device fold folds one host holding both designs' entrypoint sequences
// (kept VERBATIM -- the designs named them @configs_1/@configs_2; the toolchain
// never renames), each issuing its own config device (kept verbatim). The host
// device prints with no symbol: "main" is DeviceOp's default sym_name
// (getDefaultDeviceName()), and the assembly format elides a symbol that
// equals the default. The fold labels that host with the aiex.entry_device marker
// (a trailing dict carrying the reconfig_method), which identifies the entry
// device downstream.
// CHECK: aie.device({{.*}}) {
// CHECK: aie.runtime_sequence @configs_1
// CHECK: aiex.configure @cfg_a
// CHECK: aie.runtime_sequence @configs_2
// CHECK: aiex.configure @cfg_b
// CHECK: aiex.entry_device = {reconfig_method = "ctrlpkt"}
// CHECK: aie.device({{.*}}) @cfg_a
// CHECK: aie.device({{.*}}) @cfg_b
