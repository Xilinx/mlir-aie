// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-create-pathfinder-flows %S/Inputs/flow_order_a.mlir -o %t.a.mlir
// RUN: aie-opt --aie-create-pathfinder-flows %S/Inputs/flow_order_b.mlir -o %t.b.mlir
// RUN: diff %t.a.mlir %t.b.mlir

// Verify that routing an identical set of circuit flows is independent of
// their textual order. The two inputs differ only in aie.flow declaration
// order; their routed output must be identical.
