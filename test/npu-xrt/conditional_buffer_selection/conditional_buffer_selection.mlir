//===- conditional_buffer_selection.mlir -----------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// The below lowered code was produced from the following higher level MLIR using the following pipeline:
//
// Pipeline: 
//     builtin.module(lower-affine,aie-canonicalize-device,aie.device(aie-assign-lock-ids,aie-register-objectFifos,aie-objectFifo-stateful-transform{ dynamic-objFifos=0 },aie-assign-bd-ids,aie-lower-cascade-flows,aie-lower-broadcast-packet,aie-lower-multicast,aie-assign-tile-controller-ids,aie-generate-column-control-overlay{ route-shim-to-tile-ctrl=0 },aie-assign-buffer-addresses{ alloc-scheme=bank-aware }),convert-scf-to-cf)
//
// MLIR:
//
//     module {
//       aie.device(npu1_4col) {
// 
//         func.func private @do_something(memref<64x32xf32>)
// 
//         %tile_0_2 = aie.tile(0, 2)
//         %buf_a = aie.buffer(%tile_0_2) {sym_name = "buf_a"} : memref<64x32xf32> 
//         %buf_b = aie.buffer(%tile_0_2) {sym_name = "buf_b"} : memref<64x32xf32> 
// 
//         %core_0_2 = aie.core(%tile_0_2) {
// 
//           %idx0 = index.constant 0
//           %idx1 = index.constant 1
// 
//           %loop = scf.for %arg2 = %idx0 to %idx1 step %idx1 iter_args(%iter = %idx0) -> (index) {
// 
//             // Problematic section: we select a buffer based on a condition ...
//             %cmp_res = index.cmp eq(%iter, %idx0)
//             %branch_res = scf.if %cmp_res -> (memref<64x32xf32>) {
//               scf.yield %buf_a : memref<64x32xf32>
//             } else {
//               scf.yield %buf_b : memref<64x32xf32>
//             }
//             // ... then use this buffer in some computation.
//             func.call @do_something(%branch_res) : (memref<64x32xf32>) -> ()
// 
//             %next_iter = index.add %idx0, %idx1
//             scf.yield %next_iter : index
//           }
// 
//           aie.end
//         }
//       }
//     }
//
// Test with:
//     aie-opt --pass-pipeline="builtin.module(aie.device(aie-materialize-bd-chains,aie-substitute-shim-dma-allocations,aie-assign-runtime-sequence-bd-ids,aie-dma-tasks-to-npu,aie-dma-to-npu))" conditional_buffer_selection.mlir

module {
  aie.device(npu1_4col) {
    %tile_0_0 = aie.tile(0, 0)
    func.func private @do_something(memref<64x32xf32>)
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %buf_a = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "buf_a"} : memref<64x32xf32> 
    %buf_b = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "buf_b"} : memref<64x32xf32> 
    %core_0_2 = aie.core(%tile_0_2) {
      %idx0 = index.constant 0
      %idx1 = index.constant 1
      cf.br ^bb1(%idx0, %idx0 : index, index)
    ^bb1(%0: index, %1: index):  // 2 preds: ^bb0, ^bb6
      %2 = arith.cmpi slt, %0, %idx1 : index
      cf.cond_br %2, ^bb2, ^bb7
    ^bb2:  // pred: ^bb1
      %3 = index.cmp eq(%1, %idx0)
      cf.cond_br %3, ^bb3, ^bb4
    ^bb3:  // pred: ^bb2
      cf.br ^bb5(%buf_a : memref<64x32xf32>)
    ^bb4:  // pred: ^bb2
      cf.br ^bb5(%buf_b : memref<64x32xf32>)
    ^bb5(%4: memref<64x32xf32>):  // 2 preds: ^bb3, ^bb4
      cf.br ^bb6
    ^bb6:  // pred: ^bb5
      func.call @do_something(%4) : (memref<64x32xf32>) -> ()
      %5 = index.add %idx0, %idx1
      %6 = arith.addi %0, %idx1 : index
      cf.br ^bb1(%6, %5 : index, index)
    ^bb7:  // pred: ^bb1
      aie.end
    }
  }
}
