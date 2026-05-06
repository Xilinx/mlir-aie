//===- cpp_link_with_shared_func.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Test that two cores each calling the same func.func @kernel {link_with="k.o"}
// each produce exactly one INPUT(k.o) / _include _file k.o entry (no
// duplication of the shared object file).

// RUN: aie-opt --aie-assign-core-link-files %s | FileCheck %s --check-prefix=OPT
// RUN: aie-opt --aie-assign-core-link-files %s | aie-translate --aie-generate-ldscript --tilecol=0 --tilerow=2 | FileCheck %s --check-prefix=LDSCRIPT02
// RUN: aie-opt --aie-assign-core-link-files %s | aie-translate --aie-generate-ldscript --tilecol=0 --tilerow=3 | FileCheck %s --check-prefix=LDSCRIPT03

// OPT-COUNT-2: link_files = ["k.o"]

// LDSCRIPT02: INPUT(k.o)
// LDSCRIPT02-NOT: INPUT(k.o)

// LDSCRIPT03: INPUT(k.o)
// LDSCRIPT03-NOT: INPUT(k.o)

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)

    // Declare objectfifos before the cores that reference them.
    aie.objectfifo @dummy_in(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @dummy_in2(%tile_0_0, {%tile_0_3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    func.func private @kernel(memref<16xi32>) attributes {link_with = "k.o"}

    %core_0_2 = aie.core(%tile_0_2) {
      %buf = aie.objectfifo.acquire @dummy_in(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
      %elem = aie.objectfifo.subview.access %buf[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
      func.call @kernel(%elem) : (memref<16xi32>) -> ()
      aie.objectfifo.release @dummy_in(Consume, 1)
      aie.end
    }

    %core_0_3 = aie.core(%tile_0_3) {
      %buf = aie.objectfifo.acquire @dummy_in2(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
      %elem = aie.objectfifo.subview.access %buf[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
      func.call @kernel(%elem) : (memref<16xi32>) -> ()
      aie.objectfifo.release @dummy_in2(Consume, 1)
      aie.end
    }

    aie.runtime_sequence() {}
  }
}
