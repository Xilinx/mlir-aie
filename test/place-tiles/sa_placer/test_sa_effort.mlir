//===- test_sa_effort.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// sa-effort=1.0 is the default budget.
// RUN: aie-opt --aie-place-tiles='placer=sa_placer sa-seed=42' %s > %t.default.mlir
// RUN: aie-opt --aie-place-tiles='placer=sa_placer sa-seed=42 sa-effort=1.0' %s > %t.full.mlir
// RUN: diff %t.default.mlir %t.full.mlir

// A short budget still places deterministically.
// RUN: aie-opt --aie-place-tiles='placer=sa_placer sa-seed=42 sa-effort=0.25' %s > %t.short1.mlir
// RUN: aie-opt --aie-place-tiles='placer=sa_placer sa-seed=42 sa-effort=0.25' %s > %t.short2.mlir
// RUN: diff %t.short1.mlir %t.short2.mlir
// RUN: FileCheck %s < %t.short1.mlir

// RUN: aie-opt --aie-place-tiles='placer=sa_placer sa-seed=42 sa-effort=0.25' --mlir-pass-statistics %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=STATS

// RUN: not aie-opt --aie-place-tiles='placer=sa_placer sa-effort=0' %s 2>&1 | FileCheck %s --check-prefix=ERR

// CHECK-NOT: aie.logical_tile
// CHECK: aie.cascade_flow

// STATS: AIEPlaceTiles
// STATS: (S) {{[0-9]+}} sa-final-cost

// ERR: error: sa-effort must be positive, got 0

module @effort_4core {
  aie.device(npu2) {
    %shim1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %mem1 = aie.logical_tile<MemTile>(?, ?)
    %c0 = aie.logical_tile<CoreTile>(?, ?)
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    %c2 = aie.logical_tile<CoreTile>(?, ?)
    %c3 = aie.logical_tile<CoreTile>(?, ?)

    aie.cascade_flow(%c0, %c1)

    aie.objectfifo @in(%shim1, {%mem1}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @in0(%mem1, {%c0}, 2 : i32) : !aie.objectfifo<memref<512xi32>>
    aie.objectfifo @in1(%mem1, {%c2}, 2 : i32) : !aie.objectfifo<memref<512xi32>>
    aie.objectfifo.link [@in] -> [@in0, @in1]([] [0, 512])

    aie.objectfifo @mid(%c2, {%c3}, 2 : i32) : !aie.objectfifo<memref<512xi32>>

    aie.objectfifo @out0(%c1, {%mem1}, 2 : i32) : !aie.objectfifo<memref<512xi32>>
    aie.objectfifo @out1(%c3, {%mem1}, 2 : i32) : !aie.objectfifo<memref<512xi32>>
    aie.objectfifo @out(%mem1, {%shim1}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@out0, @out1] -> [@out]([0, 512] [])

    aie.core(%c0) { aie.end }
    aie.core(%c1) { aie.end }
    aie.core(%c2) { aie.end }
    aie.core(%c3) { aie.end }
    aie.end
  }
}
