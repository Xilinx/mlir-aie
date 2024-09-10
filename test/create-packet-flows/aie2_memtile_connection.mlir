//===- aie2_memtile_connection.mlir ----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows --aie-find-flows %s -o %t.opt
// RUN: FileCheck %s --check-prefix=CHECK1 < %t.opt
// RUN: aie-translate --aie-flows-to-json %t.opt | FileCheck %s --check-prefix=CHECK2

// CHECK1:    %[[T00:.*]] = aie.tile(0, 0)
// CHECK1:    %[[T01:.*]] = aie.tile(0, 1)
// CHECK1:    %[[T02:.*]] = aie.tile(0, 2)
// CHECK1:    aie.flow(%[[T01]], DMA : 0, %[[T00]], DMA : 0)
// CHECK1:    aie.packet_flow(0) {
// CHECK1:      aie.packet_source<%[[T02]], DMA : 0>
// CHECK1:      aie.packet_dest<%[[T00]], DMA : 1>
// CHECK1:    }

// CHECK2: "total_path_length": 3

module {
 aie.device(npu1_1col) {
  %tile_0_0 = aie.tile(0, 0)
  %tile_0_1 = aie.tile(0, 1)
  %tile_0_2 = aie.tile(0, 2)
  aie.flow(%tile_0_1, DMA : 0, %tile_0_0, DMA : 0) 
  aie.packet_flow(0) { 
    aie.packet_source<%tile_0_2, DMA : 0> 
    aie.packet_dest<%tile_0_0, DMA : 1>
  }
 }
}
