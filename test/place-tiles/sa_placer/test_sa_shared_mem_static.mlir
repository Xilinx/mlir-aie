//===- test_sa_shared_mem_static.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A shared-memory fifo between two cores must see the static buffers already
// on each tile when the SA placer picks which of the two holds its pool.
// Core A holds 60KB of weights; the 8KB pool only fits on core B.

// RUN: aie-opt --aie-place-tiles='placer=sa_placer sa-seed=1' %s | FileCheck %s
// RUN: aie-opt --aie-place-tiles='placer=sa_placer sa-seed=1' --aie-objectFifo-stateful-transform --aie-assign-buffer-addresses %s | FileCheck %s --check-prefix=ADDR

// CHECK: aie.buffer(%[[WTS:[a-z0-9_]+]]) {sym_name = "weights"}
// CHECK-NOT: aie.objectfifo.allocate @data(%[[WTS]])
// CHECK: aie.objectfifo.allocate @data(

// ADDR-DAG: {address = {{[0-9]+}} : i32, mem_bank = {{[0-9]+}} : i32, sym_name = "data_buff_0"}
// ADDR-DAG: {address = {{[0-9]+}} : i32, mem_bank = {{[0-9]+}} : i32, sym_name = "data_buff_1"}
// ADDR-DAG: {address = {{[0-9]+}} : i32, mem_bank = {{[0-9]+}} : i32, sym_name = "weights"}
module @shared_mem_pool_avoids_static_buffers {
  aie.device(npu2) {
    %coreA = aie.logical_tile<CoreTile>(?, ?)
    %coreB = aie.logical_tile<CoreTile>(?, ?)
    %wts = aie.buffer(%coreA) {sym_name = "weights"} : memref<15360xi32>
    aie.objectfifo @data(%coreA, {%coreB}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.core(%coreA) {
      %0 = aie.objectfifo.acquire @data(Produce, 1) : memref<1024xi32>
      aie.objectfifo.release @data(Produce, 1)
      aie.end
    }
    aie.core(%coreB) {
      %0 = aie.objectfifo.acquire @data(Consume, 1) : memref<1024xi32>
      aie.objectfifo.release @data(Consume, 1)
      aie.end
    }
  }
}
