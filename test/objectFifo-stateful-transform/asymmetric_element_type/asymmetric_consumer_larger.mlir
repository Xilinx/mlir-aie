//===- asymmetric_consumer_larger.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// ObjectFifo whose consumer takes larger objects (producer 10xi32, consumer
// 40xi32): the producer sends 10 elements per BD, the consumer receives 40
// (1:4), each end's DMA at its own length.

// Consumer-side buffer should be the large type
// CHECK-DAG: aie.buffer(%{{.*}}) {sym_name = "test_wts__cons__buf0"} : memref<40xi32>

// Producer-side buffer should be the small type
// CHECK-DAG: aie.buffer(%{{.*}}) {sym_name = "test_wts__buf0"} : memref<10xi32>

// Flow connecting producer DMA to consumer DMA
// CHECK-DAG: aie.flow

// Producer DMA sends 10 elements per BD
// CHECK: aie.memtile_dma
// CHECK: aie.dma_bd(%{{.*}} : memref<10xi32> offset = {{.*}} len = {{.*}})

// Consumer DMA receives 40 elements per BD
// CHECK: aie.mem
// CHECK: aie.dma_bd(%{{.*}} : memref<40xi32> offset = {{.*}} len = {{.*}})

module {
  aie.device(npu2) {
    %mt = aie.tile(0, 1)
    %ct = aie.tile(0, 2)

    aie.objectfifo @test_wts(%mt, {%ct}, 1 : i32)
      : !aie.objectfifo<memref<10xi32>>
      -> !aie.objectfifo<memref<40xi32>>

    %c = aie.core(%ct) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %elem = aie.objectfifo.acquire @test_wts(Consume, 1) : memref<40xi32>
        aie.objectfifo.release @test_wts(Consume, 1)
      }
      aie.end
    }
  }
}
