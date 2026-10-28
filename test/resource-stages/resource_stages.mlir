//===- resource_stages.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// aie-resource-stages runs aiecc's stages from placement to routing, so a
// frontend can ask whether a design fits without compiling its cores: tiles
// placed, objectFIFOs lowered to locks and buffers with addresses, and flows
// routed through switchboxes.

// RUN: aie-opt --pass-pipeline='builtin.module(aie-resource-stages)' %s | FileCheck %s
// RUN: aie-opt --pass-pipeline='builtin.module(aie-resource-stages{placer=sa_placer dynamic-objFifos=false})' %s | FileCheck %s

// CHECK: aie.device(npu1_1col)
// CHECK-DAG: %[[SHIM:.*]] = aie.tile(0, 0)
// CHECK-DAG: %[[CORE:.*]] = aie.tile(0, {{[2-5]}})
// CHECK-DAG: aie.lock(%[[CORE]], {{[0-9]+}}) {init = 2 : i32, sym_name = "in_cons_prod_lock_0"}
// CHECK-DAG: aie.buffer(%[[CORE]]) {address = {{[0-9]+}} : i32, mem_bank = {{[0-9]+}} : i32, sym_name = "in_cons_buff_0"}
// CHECK-DAG: aie.switchbox(%[[SHIM]])
// CHECK-DAG: aie.mem(%[[CORE]])

module {
  aie.device(npu1_1col) {
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    %core = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @in(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%core) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %o = aie.objectfifo.acquire @in (Consume, 1) : memref<16xi32>
        aie.objectfifo.release @in (Consume, 1)
        scf.yield
      }
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      %t = aiex.dma_configure_task_for @in {
        aie.dma_bd(%arg0 : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}
