//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// in0 starts with two objects of initial values and shares the link's pool on
// (0,3) through its allocate, so the core on (0,2) writes its objects there
// directly. The link's output drains the initial objects first, then the
// core's: object i holds 100 * i + j.

module {
  aie.device(NPUDEVICE) {
    %s = aie.tile(0, 0)
    %a = aie.tile(0, 2)
    %l = aie.tile(0, 3)

    aie.objectfifo @in0(%a, {%l}, 2 : i32) : !aie.objectfifo<memref<16xi32>> = [dense<[1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015]> : memref<16xi32>, dense<[2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015]> : memref<16xi32>]
    aie.objectfifo @out(%l, {%s}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in0] -> [@out] ([] [])
    aie.objectfifo.allocate @in0 (%l)

    aie.core(%a) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c6 = arith.constant 6 : index
      %c16 = arith.constant 16 : index
      %k100 = arith.constant 100 : i32
      scf.for %i = %c0 to %c6 step %c1 {
        %buf = aie.objectfifo.acquire @in0(Produce, 1) : memref<16xi32>
        %ii = arith.index_cast %i : index to i32
        %base = arith.muli %ii, %k100 : i32
        scf.for %j = %c0 to %c16 step %c1 {
          %jj = arith.index_cast %j : index to i32
          %v = arith.addi %base, %jj : i32
          memref.store %v, %buf[%j] : memref<16xi32>
        }
        aie.objectfifo.release @in0(Produce, 1)
      }
      aie.end
    }

    aie.runtime_sequence(%out : memref<96xi32>) {
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][1, 1, 1, 96][0, 0, 0, 1]) {metadata = @out, id = 0 : i64, issue_token = true} : memref<96xi32>
      aiex.npu.dma_wait {symbol = @out}
    }
  }
}
