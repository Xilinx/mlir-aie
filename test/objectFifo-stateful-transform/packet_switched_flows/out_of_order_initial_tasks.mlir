// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-objectfifo-split --aie-assign-packet-ids --aie-objectfifo-allocate --aie-objectfifo-insert-core-queue-pushes --aie-objectfifo-lower-dmas --aie-assign-bd-ids %s | FileCheck %s

// Five unbounded FIFOs initialize one task and use one shared counter.
// CHECK-LABEL: module @unbounded
// CHECK: memref.alloca() : memref<i32>
// CHECK-NOT: memref.alloca() : memref<i32>
// CHECK: aie.dma_start(S2MM, 0, {{.*}}, repeat_count = 255) {out_of_order}
// CHECK-NOT: aie.dma_start
module @unbounded {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @f0(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], prod_port = #aie.end_port<DMA : 0>, transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f1(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], prod_port = #aie.end_port<DMA : 0>, transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f2(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], prod_port = #aie.end_port<DMA : 0>, transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f3(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], prod_port = #aie.end_port<DMA : 0>, transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f4(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], prod_port = #aie.end_port<DMA : 0>, transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    %core_0_2 = aie.core(%tile_0_2) {
      %0 = aie.objectfifo.acquire @f0(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f0(Consume, 1)
      %1 = aie.objectfifo.acquire @f1(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f1(Consume, 1)
      %2 = aie.objectfifo.acquire @f2(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f2(Consume, 1)
      %3 = aie.objectfifo.acquire @f3(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f3(Consume, 1)
      %4 = aie.objectfifo.acquire @f4(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f4(Consume, 1)
      aie.end
    }
  }
}


// -----

// Six bounded FIFOs need twelve credits, which fit in one task.
// CHECK-LABEL: module @bounded
// CHECK-NOT: aie.dma_push_queue
// CHECK: aie.dma_start(S2MM, 0, {{.*}}, repeat_count = 11) {out_of_order}
// CHECK-NOT: aie.dma_start
module @bounded {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @f0(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], iter_count = 1 : i32, prod_port = #aie.end_port<DMA : 0>, transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f1(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], iter_count = 1 : i32, prod_port = #aie.end_port<DMA : 0>, transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f2(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], iter_count = 1 : i32, prod_port = #aie.end_port<DMA : 0>, transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f3(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], iter_count = 1 : i32, prod_port = #aie.end_port<DMA : 0>, transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f4(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], iter_count = 1 : i32, prod_port = #aie.end_port<DMA : 0>, transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f5(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], iter_count = 1 : i32, prod_port = #aie.end_port<DMA : 0>, transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    %core_0_2 = aie.core(%tile_0_2) {
      %0 = aie.objectfifo.acquire @f0(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f0(Consume, 1)
      %1 = aie.objectfifo.acquire @f0(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f0(Consume, 1)
      %2 = aie.objectfifo.acquire @f1(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f1(Consume, 1)
      %3 = aie.objectfifo.acquire @f1(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f1(Consume, 1)
      %4 = aie.objectfifo.acquire @f2(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f2(Consume, 1)
      %5 = aie.objectfifo.acquire @f2(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f2(Consume, 1)
      %6 = aie.objectfifo.acquire @f3(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f3(Consume, 1)
      %7 = aie.objectfifo.acquire @f3(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f3(Consume, 1)
      %8 = aie.objectfifo.acquire @f4(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f4(Consume, 1)
      %9 = aie.objectfifo.acquire @f4(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f4(Consume, 1)
      %10 = aie.objectfifo.acquire @f5(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f5(Consume, 1)
      %11 = aie.objectfifo.acquire @f5(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f5(Consume, 1)
      aie.end
    }
  }
}


// -----

// CHECK-LABEL: module @mixed
// CHECK: aie.objectfifo.release @f0__cons(1)
// CHECK: aie.dma_push_queue(S2MM : 0)
// CHECK: aie.objectfifo.release @f1__cons(1)
// CHECK: aie.dma_push_queue(S2MM : 0)
// CHECK: aie.dma_start(S2MM, 0, {{.*}}, repeat_count = 255) {out_of_order}
// CHECK-NOT: aie.dma_start
module @mixed {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @f0(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], prod_port = #aie.end_port<DMA : 0>, iter_count = 1 : i32, transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f1(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], prod_port = #aie.end_port<DMA : 0>, transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f2(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], prod_port = #aie.end_port<DMA : 0>, transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f3(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], prod_port = #aie.end_port<DMA : 0>, transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f4(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], prod_port = #aie.end_port<DMA : 0>, transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    %core_0_2 = aie.core(%tile_0_2) {
      %0 = aie.objectfifo.acquire @f0(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f0(Consume, 1)
      %1 = aie.objectfifo.acquire @f1(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f1(Consume, 1)
      %2 = aie.objectfifo.acquire @f2(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f2(Consume, 1)
      %3 = aie.objectfifo.acquire @f3(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f3(Consume, 1)
      %4 = aie.objectfifo.acquire @f4(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f4(Consume, 1)
      aie.end
    }
  }
}

// -----

// CHECK-LABEL: module @two_tasks
// CHECK: aie.dma_start(S2MM, 0, {{.*}}, repeat_count = 255) {out_of_order}
// CHECK: aie.dma_start(S2MM, 0, {{.*}}, repeat_count = 43) {out_of_order}
// CHECK-NOT: aie.dma_start
module @two_tasks {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @f0(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], iter_count = 25 : i32, prod_port = #aie.end_port<DMA : 0>, transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f1(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], iter_count = 25 : i32, prod_port = #aie.end_port<DMA : 0>, transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f2(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], iter_count = 25 : i32, prod_port = #aie.end_port<DMA : 0>, transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f3(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], iter_count = 25 : i32, prod_port = #aie.end_port<DMA : 0>, transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f4(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], iter_count = 25 : i32, prod_port = #aie.end_port<DMA : 0>, transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f5(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], iter_count = 25 : i32, prod_port = #aie.end_port<DMA : 0>, transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    %core_0_2 = aie.core(%tile_0_2) {
      %0 = aie.objectfifo.acquire @f0(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f0(Consume, 1)
      %1 = aie.objectfifo.acquire @f0(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f0(Consume, 1)
      %2 = aie.objectfifo.acquire @f1(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f1(Consume, 1)
      %3 = aie.objectfifo.acquire @f1(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f1(Consume, 1)
      %4 = aie.objectfifo.acquire @f2(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f2(Consume, 1)
      %5 = aie.objectfifo.acquire @f2(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f2(Consume, 1)
      %6 = aie.objectfifo.acquire @f3(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f3(Consume, 1)
      %7 = aie.objectfifo.acquire @f3(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f3(Consume, 1)
      %8 = aie.objectfifo.acquire @f4(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f4(Consume, 1)
      %9 = aie.objectfifo.acquire @f4(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f4(Consume, 1)
      %10 = aie.objectfifo.acquire @f5(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f5(Consume, 1)
      %11 = aie.objectfifo.acquire @f5(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f5(Consume, 1)
      aie.end
    }
  }
}

// -----

// CHECK-LABEL: module @full_queue
// CHECK: aie.dma_start(S2MM, 0, {{.*}}, repeat_count = 255) {out_of_order}
// CHECK: aie.dma_start(S2MM, 0, {{.*}}, repeat_count = 255) {out_of_order}
// CHECK: aie.dma_start(S2MM, 0, {{.*}}, repeat_count = 255) {out_of_order}
// CHECK: aie.dma_start(S2MM, 0, {{.*}}, repeat_count = 251) {out_of_order}
// CHECK-NOT: aie.dma_start
module @full_queue {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @f0(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], iter_count = 85 : i32, prod_port = #aie.end_port<DMA : 0>, transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f1(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], iter_count = 85 : i32, prod_port = #aie.end_port<DMA : 0>, transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f2(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], iter_count = 85 : i32, prod_port = #aie.end_port<DMA : 0>, transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f3(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], iter_count = 85 : i32, prod_port = #aie.end_port<DMA : 0>, transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f4(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], iter_count = 85 : i32, prod_port = #aie.end_port<DMA : 0>, transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f5(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], iter_count = 85 : i32, prod_port = #aie.end_port<DMA : 0>, transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    %core_0_2 = aie.core(%tile_0_2) {
      %0 = aie.objectfifo.acquire @f0(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f0(Consume, 1)
      %1 = aie.objectfifo.acquire @f0(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f0(Consume, 1)
      %2 = aie.objectfifo.acquire @f1(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f1(Consume, 1)
      %3 = aie.objectfifo.acquire @f1(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f1(Consume, 1)
      %4 = aie.objectfifo.acquire @f2(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f2(Consume, 1)
      %5 = aie.objectfifo.acquire @f2(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f2(Consume, 1)
      %6 = aie.objectfifo.acquire @f3(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f3(Consume, 1)
      %7 = aie.objectfifo.acquire @f3(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f3(Consume, 1)
      %8 = aie.objectfifo.acquire @f4(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f4(Consume, 1)
      %9 = aie.objectfifo.acquire @f4(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f4(Consume, 1)
      %10 = aie.objectfifo.acquire @f5(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f5(Consume, 1)
      %11 = aie.objectfifo.acquire @f5(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f5(Consume, 1)
      aie.end
    }
  }
}

// -----

module @overflow {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    // expected-error@+1 {{needs 5 initial tasks for 1032 objects, exceeding its queue capacity}}
    aie.objectfifo @f0(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], iter_count = 86 : i32, prod_port = #aie.end_port<DMA : 0>, transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f1(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], iter_count = 86 : i32, prod_port = #aie.end_port<DMA : 0>, transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f2(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], iter_count = 86 : i32, prod_port = #aie.end_port<DMA : 0>, transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f3(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], iter_count = 86 : i32, prod_port = #aie.end_port<DMA : 0>, transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f4(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], iter_count = 86 : i32, prod_port = #aie.end_port<DMA : 0>, transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f5(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], iter_count = 86 : i32, prod_port = #aie.end_port<DMA : 0>, transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    %core_0_2 = aie.core(%tile_0_2) {
      %0 = aie.objectfifo.acquire @f0(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f0(Consume, 1)
      %1 = aie.objectfifo.acquire @f0(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f0(Consume, 1)
      %2 = aie.objectfifo.acquire @f1(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f1(Consume, 1)
      %3 = aie.objectfifo.acquire @f1(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f1(Consume, 1)
      %4 = aie.objectfifo.acquire @f2(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f2(Consume, 1)
      %5 = aie.objectfifo.acquire @f2(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f2(Consume, 1)
      %6 = aie.objectfifo.acquire @f3(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f3(Consume, 1)
      %7 = aie.objectfifo.acquire @f3(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f3(Consume, 1)
      %8 = aie.objectfifo.acquire @f4(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f4(Consume, 1)
      %9 = aie.objectfifo.acquire @f4(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f4(Consume, 1)
      %10 = aie.objectfifo.acquire @f5(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f5(Consume, 1)
      %11 = aie.objectfifo.acquire @f5(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f5(Consume, 1)
      aie.end
    }
  }
}
