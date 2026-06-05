//===- subview_escape_via_iter_args.mlir ----------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Regression: previously the verifier crashed with SIGSEGV when an
// !aie.objectfifosubview escaped an scf.for via iter_args / scf.yield,
// because the verifier assumed the subview's defining op was always a
// direct aie.objectfifo.acquire. The verifier now rejects the construct
// with a clean diagnostic at parse/verify time — downstream lowering
// can rely on the direct-acquire invariant unconditionally.

// RUN: aie-opt --verify-diagnostics %s

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %buf = aie.buffer(%tile_0_2) {sym_name = "buf"} : memref<8xi8>

    aie.objectfifo @fifo(%tile_0_1, {%tile_0_2}, 3 : i32) : !aie.objectfifo<memref<8xi8>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c14 = arith.constant 14 : index
      %init_sv = aie.objectfifo.acquire @fifo(Consume, 1) : !aie.objectfifosubview<memref<8xi8>>
      %final_sv = scf.for %arg0 = %c0 to %c14 step %c1 iter_args(%sv_arg = %init_sv) -> (!aie.objectfifosubview<memref<8xi8>>) {
        %x = aie.objectfifo.acquire @fifo(Consume, 3) : !aie.objectfifosubview<memref<8xi8>>
        aie.objectfifo.release @fifo(Consume, 1)
        scf.yield %x : !aie.objectfifosubview<memref<8xi8>>
      }
      // expected-error@+1 {{subview operand must be the direct result of an aie.objectfifo.acquire}}
      %x0 = aie.objectfifo.subview.access %final_sv[0] : !aie.objectfifosubview<memref<8xi8>> -> memref<8xi8>
      scf.for %i = %c0 to %c8 step %c1 {
        %v = memref.load %x0[%i] : memref<8xi8>
        memref.store %v, %buf[%i] : memref<8xi8>
      }
      aie.objectfifo.release @fifo(Consume, 2)
      aie.end
    }
  }
}
