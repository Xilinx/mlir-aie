//===- edge-weight.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Edge width reports the bytes a flow group moves, on a log scale spanning the
// design's own range, with a legend naming the lightest and heaviest.

// REQUIRES: aie-visualize
// RUN: aie-visualize --emit-dot --scale=1 --contract=port,buffer %s | FileCheck %s --check-prefix=UNIFORM
// RUN: aie-visualize --emit-dot --scale=1 --contract=port,buffer --weigh-edges %s | FileCheck %s --check-prefix=WEIGHED
// RUN: aie-visualize --emit-dot --scale=1 --contract=port,buffer --weigh-edges --min-edge-width=1 --max-edge-width=3 %s | FileCheck %s --check-prefix=RANGE

// Without the option every edge keeps the same width and carries no volume.
// UNIFORM-NOT: legend_
// UNIFORM: penwidth="2.40"
// UNIFORM-NOT: per iteration

// The light flow moves 64 B and the heavy one 4096 B, so they take the ends of
// the width range and the legend names both.
// WEIGHED-DAG: legend_title {{.*}}label="bytes moved per iteration"
// WEIGHED-DAG: legend_0_text {{.*}}label="64 B"
// WEIGHED-DAG: label="4.0 KiB"
// WEIGHED-DAG: penwidth="0.50"{{.*}}tooltip="64 B per iteration"
// WEIGHED-DAG: penwidth="7.00"{{.*}}tooltip="4.0 KiB per iteration"

// The ends of the range follow the options.
// RANGE-DAG: penwidth="1.00"{{.*}}tooltip="64 B per iteration"
// RANGE-DAG: penwidth="3.00"{{.*}}tooltip="4.0 KiB per iteration"

module {
  aie.device(npu1_1col) {
    func.func private @consume(memref<16xi32>)

    %shim = aie.tile(0, 0)
    %light = aie.tile(0, 2)
    %heavy = aie.tile(0, 3)

    %small = aie.buffer(%light) : memref<16xi32>
    %large = aie.buffer(%heavy) : memref<1024xi32>

    aie.core(%light) {
      func.call @consume(%small) : (memref<16xi32>) -> ()
      aie.end
    }

    aie.mem(%light) {
      %0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%small : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }

    aie.mem(%heavy) {
      %0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%large : memref<1024xi32> offset = 0 len = 1024)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }

    aie.flow(%light, DMA : 0, %shim, DMA : 0)
      via (%light : DMA : 0 -> South : 0,
           %shim : North : 0 -> DMA : 0)
    aie.flow(%heavy, DMA : 0, %shim, DMA : 1)
      via (%heavy : DMA : 0 -> South : 1,
           %shim : North : 1 -> DMA : 1)
  }
}
