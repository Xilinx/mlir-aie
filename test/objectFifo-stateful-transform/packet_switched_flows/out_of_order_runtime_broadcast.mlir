// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-assign-bd-ids %s | FileCheck %s

// Runtime-generated headers must select the same buffer at both destinations,
// even when one tile reserves an ID before ObjectFifo allocation.
// CHECK-DAG: aie.dma_bd(%a__cons0__buf0{{.*}}{bd_id = [[PING:[0-9]+]] : i32
// CHECK-DAG: aie.dma_bd(%a__cons0__buf1{{.*}}{bd_id = [[PONG:[0-9]+]] : i32
// CHECK-DAG: aie.dma_bd(%a__cons1__buf0{{.*}}{bd_id = [[PING]] : i32
// CHECK-DAG: aie.dma_bd(%a__cons1__buf1{{.*}}{bd_id = [[PONG]] : i32
module {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %x = aie.tile(0, 2)
    %y = aie.tile(1, 2) {aiex.reserved_bd_ids = array<i32: 0>}
    aie.objectfifo @a(%shim, {%x, %y}, 2 : i32) {iter_count = 1 : i32, transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    %core_x = aie.core(%x) {
      %object_x_0 = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a(Consume, 1)
      %object_x_1 = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a(Consume, 1)
      aie.end
    }
    %core_y = aie.core(%y) {
      %object_y_0 = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a(Consume, 1)
      %object_y_1 = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a(Consume, 1)
      aie.end
    }
  }
}
