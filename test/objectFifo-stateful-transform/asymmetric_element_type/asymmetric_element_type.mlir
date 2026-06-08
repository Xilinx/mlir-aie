//===- asymmetric_element_type.mlir ------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// ObjectFifo with asymmetric element types (producer 40xi32, consumer 10xi32).
// Producer sends 40 elements per BD, consumer receives 10 at a time (4:1 ratio).

// Consumer-side buffer should be the small type
// CHECK-DAG: aie.buffer(%{{.*}}) {sym_name = "test_wts_cons_buff_0"} : memref<10xi32>

// Producer-side buffer should be the large type
// CHECK-DAG: aie.buffer(%{{.*}}) {sym_name = "test_wts_buff_0"} : memref<40xi32>

// Flow connecting producer DMA to consumer DMA
// CHECK: aie.flow

// Producer DMA sends 40 elements per BD
// CHECK: aie.memtile_dma
// CHECK: aie.dma_bd(%{{.*}} : memref<40xi32>, 0, 40)

// Consumer DMA receives 10 elements per BD
// CHECK: aie.mem
// CHECK: aie.dma_bd(%{{.*}} : memref<10xi32>, 0, 10)

module {
  aie.device(npu2) {
    %mt = aie.tile(0, 1)
    %ct = aie.tile(0, 2)

    aie.objectfifo @test_wts(%mt, {%ct}, 1 : i32)
      : !aie.objectfifo<memref<40xi32>>
      -> !aie.objectfifo<memref<10xi32>>

    %c = aie.core(%ct) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %sv = aie.objectfifo.acquire @test_wts(Consume, 1) : !aie.objectfifosubview<memref<10xi32>>
        %elem = aie.objectfifo.subview.access %sv[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
        aie.objectfifo.release @test_wts(Consume, 1)
      }
      aie.end
    }
  }
}
