// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-objectfifo-split --aie-objectfifo-verify --verify-diagnostics %s

module {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %x = aie.tile(0, 2)
    %y = aie.tile(1, 2) {aiex.reserved_bd_ids = array<i32: 0>}
    // expected-error@+1 {{out-of-order broadcast destinations require matching buffer counts and object sizes}}
    aie.objectfifo @a(%shim, {%x, %y}, [2, 2, 3]) {transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
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
