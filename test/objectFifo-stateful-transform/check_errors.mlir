//===- check_errors.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-objectFifo-stateful-transform %s

// -----

module {
  aie.device(npu1_1col) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @fifo_in(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>>
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        // expected-error@+1 {{cannot release more elements than are already acquired}}
        %1 = aie.objectfifo.acquire @fifo_in(Consume, 1) : memref<32x32xi32>
        aie.objectfifo.release @fifo_in(Consume, 1)
        aie.objectfifo.release @fifo_in(Consume, 1)
      }
      aie.end
    }
  }
}

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

// -----

// A join's shared pool has one repeat_count, so its inputs must agree, the
// same way a distribute's outputs must.
module {
  aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile23 = aie.tile(2, 3)
    %tile11 = aie.tile(1, 1)
    %tile10 = aie.tile(1, 0)
    aie.objectfifo @of0(%tile12, {%tile11}, 2 : i32) {repeat_count = 2 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of1(%tile23, {%tile11}, 2 : i32) {repeat_count = 3 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of2(%tile11, {%tile10}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    // expected-error@+1 {{repeat counts of linked object FIFOs must be equal}}
    aie.objectfifo.link [@of0, @of1] -> [@of2] ([0, 16] [])
  }
}
