//===- visualization.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// aiecc renders the routed design through visualization-ir.mlir: mem2reg
// resolves the buffers each kernel reaches, and aie-find-flows lifts the
// routed switchboxes back into flows carrying their own route. Both cores run
// the same kernel, so they share a role and a fill.

// RUN: %aiecc -n --tmpdir %t --get=visualization-ir.mlir,visualization.dot %s
// RUN: FileCheck %s --check-prefix=IR --input-file=visualization-ir.mlir
// RUN: FileCheck %s --check-prefix=DOT --input-file=visualization.dot

// The lifted flows pin their route, and no object slot is left in memory.
// IR: aie.flow
// IR-SAME: via
// IR-NOT: memref.alloca{{.*}}memref<memref<

// DOT-DAG: tile_0_2 [shape=box{{.*}}A x2\nwork
// DOT-DAG: tile_0_3 [shape=box{{.*}}A x2\nwork
// DOT-DAG: legend_title {{.*}}label="bytes moved per iteration"

module {
  aie.device(npu1_1col) {
    func.func private @work(memref<16xi32>)

    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)

    %produced = aie.buffer(%t02) : memref<16xi32>
    %consumed = aie.buffer(%t03) : memref<16xi32>

    aie.core(%t02) {
      func.call @work(%produced) : (memref<16xi32>) -> ()
      aie.end
    }

    aie.core(%t03) {
      func.call @work(%consumed) : (memref<16xi32>) -> ()
      aie.end
    }

    aie.mem(%t02) {
      %0 = aie.dma_start(MM2S, 0, ^bd, ^end)
    ^bd:
      aie.dma_bd(%produced : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^end
    ^end:
      aie.end
    }

    aie.mem(%t03) {
      %0 = aie.dma_start(S2MM, 0, ^bd, ^end)
    ^bd:
      aie.dma_bd(%consumed : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^end
    ^end:
      aie.end
    }

    aie.flow(%t02, DMA : 0, %t03, DMA : 0)
  }
}
