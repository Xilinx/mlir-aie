//===- reconfig_union_idiomatic.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The fold writes the merged module to <tmpdir>/config_union.mlir before any
// build, so we FileCheck that artifact (aiecc's exit is ignored -- the union is
// on disk regardless of whether the later build stages need peano).
// RUN: rm -rf %t && mkdir -p %t
// RUN: cd %t && aiecc --get-full-elf --reconfig-method=ctrlpkt --get npu_lowered.mlir --tmpdir=%t %S/Inputs/reconfig_idiomatic.mlir > /dev/null 2>&1 ; cat %t/config_union.mlir | FileCheck %s

// An idiomatic single-device design (a tile-bearing device, its own runtime
// sequence, no host device) is conformed: a synthesized tile-less host device
// LIFTS the entrypoint. The design's runtime_sequence name is the user's
// entrypoint name (here the default "sequence", so the ODS printer elides it and
// both the host device @main and the lifted entrypoint print bare); the config
// device is deduced as <entrypoint>_config (here @sequence_config) and the
// design's own now-internal sequence reverts to the canonical @sequence
// (referenced only by aiex.run, never dispatched). So the dispatchable kernel is
// main:sequence; @sequence_config and the inner @sequence are purely internal.
// The fold labels the host with the aiex.entrypoint marker (a trailing dict
// carrying the reconfig_method), which identifies the entry device downstream.
// CHECK: aie.device([[ARCH:.*]]) {
// CHECK-NEXT: aie.runtime_sequence(%{{.*}}: memref<4xi32>)
// CHECK:     aiex.configure @sequence_config {
// CHECK:       aiex.run @sequence(%{{.*}}) : (memref<4xi32>)
// CHECK: aiex.entrypoint = {reconfig_method = "ctrlpkt"}
// CHECK: aie.device([[ARCH]]) @sequence_config {
