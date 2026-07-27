// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-create-pathfinder-flows %S/Inputs/flow_order_a.mlir | awk '/aie.switchbox/{box=$0} /aie.connect/{print box " | " $0}' | sort > %t.a.routing
// RUN: aie-opt --aie-create-pathfinder-flows %S/Inputs/flow_order_b.mlir | awk '/aie.switchbox/{box=$0} /aie.connect/{print box " | " $0}' | sort > %t.b.routing
// RUN: diff %t.a.routing %t.b.routing

// Verify that routing an identical set of circuit flows is independent of
// their textual order. The two inputs intentionally retain different aie.flow
// operation order. Associate every generated connection with its switchbox and
// sort those pairs so semantically identical operation ordering compares equal.
