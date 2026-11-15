//===- split-wide-accesses-aie2p.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --aievec-split-load-ups-chains="wide-accesses=true" | FileCheck %s
// RUN: aie-opt %s --aievec-split-load-ups-chains | FileCheck %s --check-prefix=OFF
// RUN: aie-opt %s --aievec-split-load-ups-chains="chains=false wide-accesses=true" | FileCheck %s --check-prefix=NOCHAIN

// With wide-accesses, a 1024- or 2048-bit load or store becomes 512-bit ones,
// consecutive runs of its elements in row-major order. A value stored that
// was joined from parts (through aievec.cast) is stored part by part. The ups
// chains are split first, as chains.

// OFF-LABEL: func.func @load_store_1024
// OFF: vector.load {{.*}} vector<32xf32>
// OFF: vector.store {{.*}} vector<32xf32>

// CHECK-LABEL: func.func @load_store_1024
// CHECK-SAME: (%[[A:.*]]: memref<64xf32>, %[[I:.*]]: index)
// CHECK: %[[L0:.*]] = vector.load %[[A]][%[[I]]] : memref<64xf32>, vector<16xf32>
// CHECK: %[[I16:.*]] = arith.addi %[[I]], %[[C16:.*]] : index
// CHECK: %[[L1:.*]] = vector.load %[[A]][%[[I16]]] : memref<64xf32>, vector<16xf32>
// CHECK: %[[V:.*]] = vector.shuffle %[[L0]], %[[L1]] [0, 1, {{.*}}, 31] : vector<16xf32>, vector<16xf32>
// CHECK: %[[W:.*]] = arith.addf %[[V]], %[[V]] : vector<32xf32>
// CHECK: %[[S0:.*]] = vector.shuffle %[[W]], %[[W]] [0, 1, {{.*}}, 15] : vector<32xf32>, vector<32xf32>
// CHECK: %[[S1:.*]] = vector.shuffle %[[W]], %[[W]] [16, 17, {{.*}}, 31] : vector<32xf32>, vector<32xf32>
// CHECK: vector.store %[[S0]], %[[A]][%[[I]]] : memref<64xf32>, vector<16xf32>
// CHECK: vector.store %[[S1]], %[[A]][%{{.*}}] : memref<64xf32>, vector<16xf32>

// CHECK-LABEL: func.func @load_store_2048
// CHECK-COUNT-4: vector.load {{.*}} : memref<64xf32>, vector<16xf32>
// CHECK-NOT: vector.shuffle
// CHECK-COUNT-4: vector.store {{.*}} : memref<64xf32>, vector<16xf32>

// CHECK-LABEL: func.func @tile_2d
// CHECK: %[[T0:.*]] = vector.load %[[B:.*]][%[[C0:.*]], %[[C0]]] : memref<4x8xf32>, vector<2x8xf32>
// CHECK: %[[T1:.*]] = vector.load %[[B]][%[[C2:.*]], %[[C0]]] : memref<4x8xf32>, vector<2x8xf32>
// CHECK: vector.insert_strided_slice %[[T0]], {{.*}} offsets = [0, 0]
// CHECK: vector.insert_strided_slice %[[T1]], {{.*}} offsets = [2, 0]
// CHECK: vector.extract_strided_slice {{.*}} offsets = [0, 0], sizes = [2, 8]
// CHECK: vector.extract_strided_slice {{.*}} offsets = [2, 0], sizes = [2, 8]
// CHECK: vector.store {{.*}}, %[[B]][%[[C0]], %[[C0]]] : memref<4x8xf32>, vector<2x8xf32>
// CHECK: vector.store {{.*}}, %[[B]][%[[C2]], %[[C0]]] : memref<4x8xf32>, vector<2x8xf32>

// CHECK-LABEL: func.func @ups_store
// CHECK: %[[U0:.*]] = aievec.ups {{.*}} : vector<32xi16>, vector<32xi32>
// CHECK: %[[U1:.*]] = aievec.ups {{.*}} : vector<32xi16>, vector<32xi32>
// CHECK-NOT: vector<64xi32>
// CHECK-COUNT-4: aievec.cast {{.*}} : vector<16xi32>, vector<16xi32>
// CHECK-COUNT-4: vector.store {{.*}} : memref<64xi32>, vector<16xi32>

// CHECK-LABEL: func.func @row_1x32
// CHECK: vector.load %[[R:.*]][%[[C0:.*]], %[[C0]]] : memref<1x32xf32>, vector<1x16xf32>
// CHECK: vector.load %[[R]][%[[C0]], %[[C16:.*]]] : memref<1x32xf32>, vector<1x16xf32>
// CHECK: vector.store {{.*}}, %[[R]][%[[C0]], %[[C0]]] : memref<1x32xf32>, vector<1x16xf32>
// CHECK: vector.store {{.*}}, %[[R]][%[[C0]], %[[C16]]] : memref<1x32xf32>, vector<1x16xf32>

// CHECK-LABEL: func.func @bf16_64
// CHECK: %[[H0:.*]] = vector.load {{.*}} : memref<64xbf16>, vector<32xbf16>
// CHECK: %[[H1:.*]] = vector.load {{.*}} : memref<64xbf16>, vector<32xbf16>
// CHECK: vector.shuffle %[[H0]], %[[H1]]

// A 2048-bit 2x32 tile: four 1x16 runs, two a row.
// CHECK-LABEL: func.func @rows_2x32
// CHECK: vector.load %[[M:.*]][%[[C0:.*]], %[[C0]]] : memref<2x32xf32>, vector<1x16xf32>
// CHECK: vector.load %[[M]][%[[C0]], %[[C16:.*]]] : memref<2x32xf32>, vector<1x16xf32>
// CHECK: vector.load %[[M]][%[[C1:.*]], %[[C0]]] : memref<2x32xf32>, vector<1x16xf32>
// CHECK: vector.load %[[M]][%[[C1]], %[[C16]]] : memref<2x32xf32>, vector<1x16xf32>
// CHECK-NOT: vector.load

// With the chains, the ups chain's halves go to two ups; without, the load is
// split and joined as any other.
// CHECK-LABEL: func.func @ups_chain
// CHECK: %[[P0:.*]] = vector.load {{.*}} : memref<64xi16>, vector<32xi16>
// CHECK: %[[P1:.*]] = vector.load {{.*}} : memref<64xi16>, vector<32xi16>
// CHECK: aievec.ups %[[P0]] {{.*}} : vector<32xi16>, vector<32xi32>
// CHECK: aievec.ups %[[P1]] {{.*}} : vector<32xi16>, vector<32xi32>
// NOCHAIN-LABEL: func.func @ups_chain
// NOCHAIN: %[[J:.*]] = vector.shuffle {{.*}} : vector<32xi16>, vector<32xi16>
// NOCHAIN: aievec.ups %[[J]] {{.*}} : vector<64xi16>, vector<64xi32>

// A cast that changes the element type keeps it on each part.
// CHECK-LABEL: func.func @ups_cast_to_f32
// CHECK-NOT: vector<64x
// CHECK-COUNT-4: aievec.cast {{.*}} : vector<16xi32>, vector<16xf32>
// CHECK-COUNT-4: vector.store {{.*}} : memref<64xf32>, vector<16xf32>

// CHECK-LABEL: func.func @narrow
// CHECK: vector.load {{.*}} : memref<64xf32>, vector<16xf32>
// CHECK-NOT: vector.load

module {
  func.func @load_store_1024(%a: memref<64xf32>, %i: index) {
    %v = vector.load %a[%i] : memref<64xf32>, vector<32xf32>
    %w = arith.addf %v, %v : vector<32xf32>
    vector.store %w, %a[%i] : memref<64xf32>, vector<32xf32>
    return
  }
  func.func @load_store_2048(%a: memref<64xf32>) {
    %c0 = arith.constant 0 : index
    %v = vector.load %a[%c0] : memref<64xf32>, vector<64xf32>
    vector.store %v, %a[%c0] : memref<64xf32>, vector<64xf32>
    return
  }
  func.func @tile_2d(%b: memref<4x8xf32>) {
    %c0 = arith.constant 0 : index
    %v = vector.load %b[%c0, %c0] : memref<4x8xf32>, vector<4x8xf32>
    %w = arith.addf %v, %v : vector<4x8xf32>
    vector.store %w, %b[%c0, %c0] : memref<4x8xf32>, vector<4x8xf32>
    return
  }
  func.func @ups_store(%in: memref<64xi16>, %out: memref<64xi32>) {
    %c0 = arith.constant 0 : index
    %v = vector.load %in[%c0] : memref<64xi16>, vector<64xi16>
    %u = aievec.ups %v {shift = 0 : i8} : vector<64xi16>, vector<64xi32>
    %c = aievec.cast %u {isResAcc = false} : vector<64xi32>, vector<64xi32>
    vector.store %c, %out[%c0] : memref<64xi32>, vector<64xi32>
    return
  }
  func.func @row_1x32(%a: memref<1x32xf32>) {
    %c0 = arith.constant 0 : index
    %v = vector.load %a[%c0, %c0] : memref<1x32xf32>, vector<1x32xf32>
    %w = arith.addf %v, %v : vector<1x32xf32>
    vector.store %w, %a[%c0, %c0] : memref<1x32xf32>, vector<1x32xf32>
    return
  }
  func.func @bf16_64(%a: memref<64xbf16>) -> vector<64xbf16> {
    %c0 = arith.constant 0 : index
    %v = vector.load %a[%c0] : memref<64xbf16>, vector<64xbf16>
    return %v : vector<64xbf16>
  }
  func.func @rows_2x32(%a: memref<2x32xf32>) -> vector<2x32xf32> {
    %c0 = arith.constant 0 : index
    %v = vector.load %a[%c0, %c0] : memref<2x32xf32>, vector<2x32xf32>
    return %v : vector<2x32xf32>
  }
  func.func @ups_chain(%in: memref<64xi16>) -> vector<64xi32> {
    %c0 = arith.constant 0 : index
    %v = vector.load %in[%c0] : memref<64xi16>, vector<64xi16>
    %u = aievec.ups %v {shift = 0 : i8} : vector<64xi16>, vector<64xi32>
    return %u : vector<64xi32>
  }
  func.func @ups_cast_to_f32(%in: memref<64xi16>, %out: memref<64xf32>) {
    %c0 = arith.constant 0 : index
    %v = vector.load %in[%c0] : memref<64xi16>, vector<64xi16>
    %u = aievec.ups %v {shift = 0 : i8} : vector<64xi16>, vector<64xi32>
    %c = aievec.cast %u {isResAcc = false} : vector<64xi32>, vector<64xf32>
    vector.store %c, %out[%c0] : memref<64xf32>, vector<64xf32>
    return
  }
  func.func @narrow(%a: memref<64xf32>) {
    %c0 = arith.constant 0 : index
    %v = vector.load %a[%c0] : memref<64xf32>, vector<16xf32>
    vector.store %v, %a[%c0] : memref<64xf32>, vector<16xf32>
    return
  }
}
