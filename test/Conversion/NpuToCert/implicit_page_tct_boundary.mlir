//===- implicit_page_tct_boundary.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implicit page formation must respect G-tct (at most one job per page may
// wait_tcts on a given (tile, channel)). Two top-level jobs waiting on the same
// actor are legal on their own -- pages run in sequence -- but grouping them
// into one implicit page makes them concurrent, which used to turn valid input
// into a verifier error. The run is cut at the collision instead.

// RUN: aie-opt -cert-legalize-pages --aie-cert-verify %s | FileCheck %s

// Colliding jobs land on separate pages...
// CHECK: aiex.cert.page
// CHECK: aiex.cert.job(1)
// CHECK: aiex.cert.wait_tcts(0, 6, 1)
// CHECK: aiex.cert.page
// CHECK: aiex.cert.job(2)
// CHECK: aiex.cert.wait_tcts(0, 6, 1)
// ...while a job waiting on a different actor still shares the page it follows.
// CHECK: aiex.cert.job(3)
// CHECK: aiex.cert.wait_tcts(0, 7, 1)
// CHECK-NOT: aiex.cert.page

module {
  aie.device(npu2) {
    aiex.cert.job(1) { aiex.cert.wait_tcts(0, 6, 1) }
    aiex.cert.job(2) { aiex.cert.wait_tcts(0, 6, 1) }
    aiex.cert.job(3) { aiex.cert.wait_tcts(0, 7, 1) }
  }
}
