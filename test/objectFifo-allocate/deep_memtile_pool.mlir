// RUN: aie-opt --split-input-file --aie-objectfifo-split --aie-objectfifo-verify --aie-assign-packet-ids --aie-objectfifo-allocate %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-objectFifo-stateful-transform %s | FileCheck %s --check-prefix=BDS

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A mem tile whose pools need more buffer descriptors than its 48 holds its
// deepest pool's objects side by side in one buffer, so each end walks them
// with one descriptor stepping an object a run: a whole 40-row map waits in
// the mem tile on two descriptors instead of eighty.

// CHECK-LABEL: @deep
// CHECK: aie.objectfifo.pool @in_cons_pool(%{{.*}}) {buffers = [@in_cons_buff_0], contiguous, depth = 40 : i32
// BDS-LABEL: @deep
// BDS: aie.memtile_dma
// BDS-COUNT-2: iteration = #aie.bd_iteration<size = 40, stride = 640, current = 0>
// BDS-NOT: aie.dma_bd
// BDS: aie.end
module @deep {
  aie.device(npu2_1col) {
    %s = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    %t = aie.tile(0, 2)
    aie.objectfifo @in(%s, {%m}, [2 : i32, 40 : i32]) : !aie.objectfifo<memref<640xi32>>
    aie.objectfifo @held(%m, {%t}, 2 : i32) : !aie.objectfifo<memref<640xi32>>
    aie.objectfifo.link [@in] -> [@held] ([] [])
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c40 = arith.constant 40 : index
      scf.for %i = %c0 to %c40 step %c1 {
        %x = aie.objectfifo.acquire @held(Consume, 1) : memref<640xi32>
        aie.objectfifo.release @held(Consume, 1)
      }
      aie.end
    }
  }
}

// -----

// Pools whose descriptors fit keep an object a buffer.
// CHECK-LABEL: @shallow
// CHECK: aie.objectfifo.pool @in_cons_pool(%{{.*}}) {buffers = [{{.*}}@in_cons_buff_7], depth = 8 : i32
module @shallow {
  aie.device(npu2_1col) {
    %s = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    %t = aie.tile(0, 2)
    aie.objectfifo @in(%s, {%m}, [2 : i32, 8 : i32]) : !aie.objectfifo<memref<640xi32>>
    aie.objectfifo @held(%m, {%t}, 2 : i32) : !aie.objectfifo<memref<640xi32>>
    aie.objectfifo.link [@in] -> [@held] ([] [])
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.for %i = %c0 to %c8 step %c1 {
        %x = aie.objectfifo.acquire @held(Consume, 1) : memref<640xi32>
        aie.objectfifo.release @held(Consume, 1)
      }
      aie.end
    }
  }
}
