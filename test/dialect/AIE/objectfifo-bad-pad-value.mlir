//===- objectfifo-bad-pad-value.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics %s

// A nonzero padValue only fills the region created by padDimensions; on its own
// it would silently no-op, so the verifier requires padDimensions be present.

aie.device(npu2_1col) {
  %shim = aie.tile(0, 0)
  %mem = aie.tile(0, 1)
  // expected-error@+1 {{`padValue` requires `padDimensions`}}
  aie.objectfifo @of(%mem, {%shim}, 2 : i32) {padValue = 7 : i32} : !aie.objectfifo<memref<64xi8>>
}
