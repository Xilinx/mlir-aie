//===- check_errors.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-objectFifo-stateful-transform %s

// -----

// An unplaced logical_tile producer must be diagnosed, not crash the pass.
module {
  aie.device(npu1) {
    %prod = aie.logical_tile<ShimNOCTile>(0, ?)
    %cons = aie.tile(0, 2)
    // expected-error@+1 {{producer tile is not a placed aie.tile; run --aie-place-tiles before this pass}}
    aie.objectfifo @of(%prod, {%cons}, 2 : i32) : !aie.objectfifo<memref<64xi16>>
  }
}

// -----

// An unplaced logical_tile consumer must be diagnosed, not crash the pass.
module {
  aie.device(npu1) {
    %prod = aie.tile(0, 0)
    %cons = aie.logical_tile<CoreTile>(?, ?)
    // expected-error@+1 {{consumer tile is not a placed aie.tile; run --aie-place-tiles before this pass}}
    aie.objectfifo @of(%prod, {%cons}, 2 : i32) : !aie.objectfifo<memref<64xi16>>
  }
}
