//===- objectfifo_allocate_link_no_link_point.mlir -------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics %s

// @f's allocate is verified before its links. The second link makes @f a
// misfit of its pool (another object type), but the first link naming @f has
// no link point, which its own verifier reports.

module {
  aie.device(npu2_1col) {
    %s = aie.tile(0, 0)
    %a = aie.tile(0, 2)
    %l = aie.tile(0, 3)
    %x = aie.tile(0, 4)
    aie.objectfifo @h(%s, {%a}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @f(%a, {%l}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @g(%x, {%s}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.allocate @f (%l)
    // expected-error@+1 {{ObjectFifoLinkOp must have a link point, i.e., a shared tile between objectFifos}}
    aie.objectfifo.link [@f] -> [@g] ([] [])
    aie.objectfifo.link [@h] -> [@f] ([] [])
  }
}
