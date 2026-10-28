//===- tile-roles.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A tile headline names the kernels it runs, without their content-hash
// prefixes and with repeats counted, and marks tiles that run the same set.

// REQUIRES: aie-visualize
// RUN: aie-visualize --emit-dot %s | FileCheck %s --check-prefix=PLAIN
// RUN: aie-visualize --emit-dot --tile-roles %s | FileCheck %s --check-prefix=ROLES
// RUN: aie-visualize --emit-dot --tile-roles --max-headline-kernels=1 %s | FileCheck %s --check-prefix=CAPPED

// Without the option a tile is labeled by coordinate alone and keeps the
// fill of its kind.
// PLAIN: tile_0_2 [shape=box{{.*}}label="(0,2)"
// PLAIN-SAME: fillcolor="#eef7ee"
// PLAIN-NOT: labelloc="t"

// The two tiles running matmul twice plus zero share a role, so both carry the
// same class marker and the same fill. The odd tile out keeps its kind fill
// and carries no marker. A headline opens with a blank line, which keeps it
// off the tile's top border.
// ROLES: tile_0_2 [shape=box{{.*}}label="\n(0,2)  A x2\nmatmul x2\nzero"
// ROLES-SAME: labelloc="t"
// ROLES-SAME: fillcolor="#cfe3f7"
// ROLES: tile_0_3 [shape=box{{.*}}label="\n(0,3)  A x2\nmatmul x2\nzero"
// ROLES-SAME: fillcolor="#cfe3f7"
// ROLES: tile_0_4 [shape=box{{.*}}label="\n(0,4)\nreduce"
// ROLES-SAME: fillcolor="#eef7ee"
// ROLES-NOT: tile_0_4 {{.*}}x2

// A headline longer than the cap counts the kernels it does not name.
// CAPPED: tile_0_2 [shape=box{{.*}}label="\n(0,2)  A x2\nmatmul x2\n+1 more"

module {
  aie.device(npu1_1col) {
    func.func private @abc12345_matmul(memref<16xi32>)
    func.func private @abc12345_zero(memref<16xi32>)
    func.func private @reduce(memref<16xi32>)

    %first = aie.tile(0, 2)
    %second = aie.tile(0, 3)
    %third = aie.tile(0, 4)

    %a = aie.buffer(%first) : memref<16xi32>
    %b = aie.buffer(%second) : memref<16xi32>
    %c = aie.buffer(%third) : memref<16xi32>

    aie.core(%first) {
      func.call @abc12345_matmul(%a) : (memref<16xi32>) -> ()
      func.call @abc12345_matmul(%a) : (memref<16xi32>) -> ()
      func.call @abc12345_zero(%a) : (memref<16xi32>) -> ()
      aie.end
    }

    aie.core(%second) {
      func.call @abc12345_matmul(%b) : (memref<16xi32>) -> ()
      func.call @abc12345_matmul(%b) : (memref<16xi32>) -> ()
      func.call @abc12345_zero(%b) : (memref<16xi32>) -> ()
      aie.end
    }

    aie.core(%third) {
      func.call @reduce(%c) : (memref<16xi32>) -> ()
      aie.end
    }

    aie.flow(%first, Core : 0, %third, DMA : 0)
      via (%first : Core : 0 -> North : 0,
           %second : South : 0 -> North : 0,
           %third : South : 0 -> DMA : 0)
  }
}
