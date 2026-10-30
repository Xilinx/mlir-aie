//===- shim_sharing_two_devices.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// One pass instance allocates every device in the module. The first device's
// shim ends share a channel; the second, with nothing to share, is allocated
// as it would be alone, never reading the first's ends, which its lowering
// has already erased.

// CHECK-LABEL: aie.device(npu2_1col) @first
// CHECK-DAG:     aie.shim_dma_allocation @in_a_shim_alloc(%{{.*}}, MM2S, 0, <pkt_id = {{[0-9]+}}>)
// CHECK-DAG:     aie.shim_dma_allocation @in_b_shim_alloc(%{{.*}}, MM2S, 0, <pkt_id = {{[0-9]+}}>)
// CHECK-DAG:     aie.shim_dma_allocation @in_c_shim_alloc(%{{.*}}, MM2S, 1)
// CHECK-LABEL: aie.device(npu2_1col) @second
// CHECK:         aie.flow(%{{.*}}, DMA : 0, %{{.*}}, DMA : 0)
// CHECK-NOT:     packet_flow
module {
  aie.device(npu2_1col) @first {
    %s = aie.tile(0, 0)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 3)
    %c = aie.tile(0, 4)
    aie.objectfifo @in_a(%s, {%a}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in_b(%s, {%b}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in_c(%s, {%c}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%a) {
      %x = aie.objectfifo.acquire @in_a(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @in_a(Consume, 1)
      aie.end
    }
    aie.core(%b) {
      %x = aie.objectfifo.acquire @in_b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @in_b(Consume, 1)
      aie.end
    }
    aie.core(%c) {
      %x = aie.objectfifo.acquire @in_c(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @in_c(Consume, 1)
      aie.end
    }
    aie.runtime_sequence @run_a(%in : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @in_a, id = 0 : i64} : memref<16xi32>
    }
    aie.runtime_sequence @run_b(%in : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @in_b, id = 0 : i64} : memref<16xi32>
    }
    aie.runtime_sequence @run_c(%in : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @in_c, id = 0 : i64} : memref<16xi32>
    }
  }
  aie.device(npu2_1col) @second {
    %p = aie.tile(0, 2)
    %q = aie.tile(0, 5)
    %flag = aie.buffer(%q) {sym_name = "flag"} : memref<1xi32>
    aie.objectfifo @fifo(%p, {%q}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%p) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      %v = arith.constant 11 : i32
      scf.for %i = %c0 to %c3 step %c1 {
        %e = aie.objectfifo.acquire @fifo(Produce, 1) : memref<16xi32>
        memref.store %v, %e[%c0] : memref<16xi32>
        aie.objectfifo.release @fifo(Produce, 1)
      }
      aie.end
    }
    aie.core(%q) {
      %c0 = arith.constant 0 : index
      %zero = arith.constant 0 : i32
      %fv = memref.load %flag[%c0] : memref<1xi32>
      %cond = arith.cmpi sgt, %fv, %zero : i32
      %e0, %e1 = aie.objectfifo.acquire @fifo(Consume, 2) : memref<16xi32>, memref<16xi32>
      scf.if %cond {
        %t = aie.objectfifo.acquire @fifo(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @fifo(Consume, 1)
      } else {
        %f = aie.objectfifo.acquire @fifo(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @fifo(Consume, 1)
      }
      aie.objectfifo.release @fifo(Consume, 1)
      aie.end
    }
  }
}
