// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-create-pathfinder-flows %S/Inputs/flow_order_a.mlir | grep -E 'aie\.(switchbox|connect)' > %t.a.routing
// RUN: aie-opt --aie-create-pathfinder-flows %S/Inputs/flow_order_b.mlir | grep -E 'aie\.(switchbox|connect)' > %t.b.routing
// RUN: diff %t.a.routing %t.b.routing

// Verify that routing an identical set of circuit flows is independent of
// their textual order. The two inputs intentionally retain different aie.flow
// operation order, so compare only the generated physical routing operations.
