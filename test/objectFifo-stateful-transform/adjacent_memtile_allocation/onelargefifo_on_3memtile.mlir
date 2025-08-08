//===- global_generation_test.mlir ------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-DAG:     memref.global "public" @out1_cons : memref<8xi32>
// CHECK-DAG:     memref.global "public" @out1 : memref<8xi32>
// CHECK-DAG:     memref.global "public" @out0_cons : memref<8xi32>
// CHECK-DAG:     memref.global "public" @out0 : memref<8xi32>
// CHECK-DAG:     memref.global "public" @in1_cons : memref<8xi32>
// CHECK-DAG:     memref.global "public" @in1 : memref<8xi32>
// CHECK-DAG:     memref.global "public" @in0_cons : memref<96000xi32>
// CHECK-DAG:     memref.global "public" @in0 : memref<96000xi32>
// CHECK-DAG:     %[[SHIM_NOC_TILE_1_0:.*]] = aie.tile(1, 0)
// CHECK-DAG:     %[[MEM_TILE_1_1:.*]] = aie.tile(1, 1)
// CHECK-DAG:     %[[TILE_1_2:.*]] = aie.tile(1, 2)
// CHECK-DAG:     %[[IN0_CONS_BUFF_0:.*]] = aie.buffer(%[[MEM_TILE_1_1]]) {sym_name = "in0_cons_buff_0"} : memref<96000xi32>
// CHECK-DAG:     %[[IN0_CONS_BUFF_1:.*]] = aie.buffer(%{{.*}}) {sym_name = "in0_cons_buff_1"} : memref<96000xi32>
// CHECK-DAG:     %[[IN0_CONS_BUFF_2:.*]] = aie.buffer(%{{.*}}) {sym_name = "in0_cons_buff_2"} : memref<96000xi32>
// CHECK-DAG:     %[[IN0_CONS_PROD_LOCK:.*]] = aie.lock(%[[MEM_TILE_1_1]], 0) {init = 3 : i32, sym_name = "in0_cons_prod_lock_0"}
// CHECK-DAG:     %[[IN0_CONS_CONS_LOCK:.*]] = aie.lock(%[[MEM_TILE_1_1]], 1) {init = 0 : i32, sym_name = "in0_cons_cons_lock_0"}
// CHECK-DAG:     %[[IN0_PROD_LOCK:.*]] = aie.lock(%[[SHIM_NOC_TILE_1_0]], 0) {init = 1 : i32, sym_name = "in0_prod_lock_0"}
// CHECK-DAG:     %[[IN0_CONS_LOCK:.*]] = aie.lock(%[[SHIM_NOC_TILE_1_0]], 1) {init = 0 : i32, sym_name = "in0_cons_lock_0"}
// CHECK-DAG:     %[[IN1_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "in1_cons_buff_0"} : memref<8xi32>
// CHECK-DAG:     %[[IN1_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "in1_cons_buff_1"} : memref<8xi32>
// CHECK-DAG:     %[[IN1_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_1_2]], 0) {init = 2 : i32, sym_name = "in1_cons_prod_lock_0"}
// CHECK-DAG:     %[[IN1_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_1_2]], 1) {init = 0 : i32, sym_name = "in1_cons_cons_lock_0"}
// CHECK-DAG:     %[[OUT1_CONS_BUFF_0:.*]] = aie.buffer(%[[MEM_TILE_1_1]]) {sym_name = "out1_cons_buff_0"} : memref<8xi32>
// CHECK-DAG:     %[[OUT1_CONS_BUFF_1:.*]] = aie.buffer(%[[MEM_TILE_1_1]]) {sym_name = "out1_cons_buff_1"} : memref<8xi32>
// CHECK-DAG:     %[[OUT1_CONS_PROD_LOCK:.*]] = aie.lock(%[[MEM_TILE_1_1]], 2) {init = 2 : i32, sym_name = "out1_cons_prod_lock_0"}
// CHECK-DAG:     %[[OUT1_CONS_CONS_LOCK:.*]] = aie.lock(%[[MEM_TILE_1_1]], 3) {init = 0 : i32, sym_name = "out1_cons_cons_lock_0"}
// CHECK-DAG:     %[[OUT1_BUFF_0:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "out1_buff_0"} : memref<8xi32>
// CHECK-DAG:     %[[OUT1_BUFF_1:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "out1_buff_1"} : memref<8xi32>
// CHECK-DAG:     %[[OUT1_PROD_LOCK:.*]] = aie.lock(%[[TILE_1_2]], 2) {init = 2 : i32, sym_name = "out1_prod_lock_0"}
// CHECK-DAG:     %[[OUT1_CONS_LOCK:.*]] = aie.lock(%[[TILE_1_2]], 3) {init = 0 : i32, sym_name = "out1_cons_lock_0"}
// CHECK-DAG:     %[[OUT0_CONS_PROD_LOCK:.*]] = aie.lock(%[[SHIM_NOC_TILE_1_0]], 2) {init = 1 : i32, sym_name = "out0_cons_prod_lock_0"}
// CHECK-DAG:     %[[OUT0_CONS_CONS_LOCK:.*]] = aie.lock(%[[SHIM_NOC_TILE_1_0]], 3) {init = 0 : i32, sym_name = "out0_cons_cons_lock_0"}

module {
  aie.device(npu1) {
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %mem_tile_1_1 = aie.tile(1, 1)
    %tile_1_2 = aie.tile(1, 2)
    aie.objectfifo @in0(%shim_noc_tile_1_0, {%mem_tile_1_1}, 3 : i32) : !aie.objectfifo<memref<96000xi32>> 
    aie.objectfifo @in1(%mem_tile_1_1, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<8xi32>> 
    aie.objectfifo.link [@in0] -> [@in1]([] [])
    aie.objectfifo @out0(%mem_tile_1_1, {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<8xi32>> 
    aie.objectfifo @out1(%tile_1_2, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<8xi32>> 
    aie.objectfifo.link [@out1] -> [@out0]([] [])
  }
}



