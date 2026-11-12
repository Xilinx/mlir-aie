// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: rm -rf %t && mkdir -p %t && cd %t && %aiecc --load-pdi-to-ctrl-pkt --get=input_physical.mlir %s
// RUN: FileCheck %s < %t/input_physical.mlir

// CHECK-LABEL: aie.device(npu2) @cfg_add
// CHECK: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0(%{{.*}}, MM2S, 0)
// CHECK: aie.shim_dma_allocation @a__shim_alloc(%{{.*}}, MM2S, 0, <pkt_id = {{[0-9]+}}>)
// CHECK: aie.shim_dma_allocation @b__shim_alloc(%{{.*}}, MM2S, 1)

module {
  aie.device(npu2) @main {
    aie.runtime_sequence @sequence(%a : memref<16xi32>, %b : memref<16xi32>, %c : memref<16xi32>) {
      aiex.configure @flows {
      }
      aiex.configure @cfg_add {
        aiex.run @run(%a, %b, %c) : (memref<16xi32>, memref<16xi32>, memref<16xi32>)
      }
    }
  }

  aie.device(npu2) @flows {
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    aie.flow(%t01, DMA : 0, %t02, DMA : 0)
    aie.flow(%t01, DMA : 1, %t02, DMA : 1)
    aie.flow(%t01, DMA : 2, %t03, DMA : 0)
    aie.flow(%t01, DMA : 3, %t03, DMA : 1)
    aie.flow(%t01, DMA : 4, %t04, DMA : 0)
    aie.flow(%t01, DMA : 5, %t04, DMA : 1)
  }

  aie.device(npu2) @cfg_add {
    %shim = aie.tile(0, 0)
    %comp = aie.tile(0, 2)
    aie.objectfifo @a(%shim, {%comp}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%shim, {%comp}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c(%comp, {%shim}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%comp) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 16 : index
      %true = arith.constant true
      scf.while : () -> () {
        scf.condition(%true)
      } do {
        %ea = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
        %eb = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
        %ec = aie.objectfifo.acquire @c(Produce, 1) : memref<16xi32>
        scf.for %k = %c0 to %cn step %c1 {
          %va = memref.load %ea[%k] : memref<16xi32>
          %vb = memref.load %eb[%k] : memref<16xi32>
          %r = arith.addi %va, %vb : i32
          memref.store %r, %ec[%k] : memref<16xi32>
        }
        aie.objectfifo.release @a(Consume, 1)
        aie.objectfifo.release @b(Consume, 1)
        aie.objectfifo.release @c(Produce, 1)
        scf.yield
      }
      aie.end
    }
    aie.runtime_sequence @run(%a : memref<16xi32>, %b : memref<16xi32>, %c : memref<16xi32>) {
      %t_a = aiex.dma_configure_task_for @a {
        aie.dma_bd(%a : memref<16xi32> offset = 0 len = 16)
        aie.end
      }
      %t_b = aiex.dma_configure_task_for @b {
        aie.dma_bd(%b : memref<16xi32> offset = 0 len = 16)
        aie.end
      }
      %t_c = aiex.dma_configure_task_for @c {
        aie.dma_bd(%c : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_a)
      aiex.dma_start_task(%t_b)
      aiex.dma_start_task(%t_c)
      aiex.dma_await_task(%t_c)
      aiex.dma_free_task(%t_a)
      aiex.dma_free_task(%t_b)
      aiex.dma_free_task(%t_c)
    }
  }

}
