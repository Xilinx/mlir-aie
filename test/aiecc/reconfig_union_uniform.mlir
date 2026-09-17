//===- reconfig_union_uniform.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aiecc --get-full-elf --reconfig-method=ctrlpkt %S/Inputs/reconfig_uniform_a.mlir %S/Inputs/reconfig_uniform_b.mlir 2>&1 | FileCheck %s

// The reconfiguration examples emit every config from ONE template, so both
// configs name their host runtime sequence with the default `sequence`. Folded
// into one host device those names collide. Since the unique-entrypoint-naming
// change the toolchain HARD-FAILS on the collision -- each design must name its
// runtime sequence uniquely (e.g. a distinct @iron.jit(name=) per design) so
// its dispatch entrypoint main:<name> is unambiguous -- instead of silently
// auto-renaming colliding entries to config_1..N. Sibling
// reconfig_union_twodevice.mlir covers the already-distinct (verbatim) case.
// CHECK: multiple designs share the host runtime sequence name 'sequence'
