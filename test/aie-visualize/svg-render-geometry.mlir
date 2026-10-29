//===- svg-render-geometry.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: aie-visualize
// RUN: aie-visualize --emit-dot --arrows=avoidance --contract=port %s -o %t.dot
// RUN: aie-visualize --emit-json --arrows=avoidance --contract=port %s -o %t.json
// RUN: FileCheck %s --check-prefix=OWNERSHIP < %t.json
// RUN: neato -Tsvg %t.dot -o %t.svg
// RUN: %python %S/Inputs/check-svg-render-geometry.py --json %t.json %t.svg | FileCheck %s --check-prefix=CLEAN
// RUN: not %python %S/Inputs/check-svg-render-geometry.py %S/Inputs/render-geometry-diagonal.svg 2>&1 | FileCheck %s --check-prefix=DIAGONAL
// RUN: not %python %S/Inputs/check-svg-render-geometry.py %S/Inputs/render-geometry-foreign-tile.svg 2>&1 | FileCheck %s --check-prefix=FOREIGN-TILE
// RUN: not %python %S/Inputs/check-svg-render-geometry.py %S/Inputs/render-geometry-node.svg 2>&1 | FileCheck %s --check-prefix=NODE
// RUN: not %python %S/Inputs/check-svg-render-geometry.py %S/Inputs/render-geometry-overlap.svg 2>&1 | FileCheck %s --check-prefix=OVERLAP
// RUN: not %python %S/Inputs/check-svg-render-geometry.py %S/Inputs/render-geometry-narrow-gap.svg 2>&1 | FileCheck %s --check-prefix=NARROW
// RUN: not %python %S/Inputs/check-svg-render-geometry.py %S/Inputs/render-geometry-crossing.svg 2>&1 | FileCheck %s --check-prefix=CROSSING
// RUN: %python %S/Inputs/check-svg-render-geometry.py --allow-defects --json %S/Inputs/render-geometry-endpoints.json %S/Inputs/render-geometry-defects.svg | FileCheck %s --check-prefix=JSON-OWNERSHIP

// CLEAN: edges={{[1-9][0-9]*}} diagonals=0 curved_bends={{[0-9]+}} foreign_tile_crossings=0 node_intersections=0 endpoint_gaps=0 collinear_overlaps=0 narrow_gaps=0 crossings=0
// OWNERSHIP: "id": "buffer_0"
// OWNERSHIP: "kind": "buffer"
// OWNERSHIP: "tile": [
// OWNERSHIP-NEXT: 0,
// OWNERSHIP-NEXT: 2
// OWNERSHIP: "id": "call_0"
// OWNERSHIP: "kind": "call"
// OWNERSHIP: "tile": [
// OWNERSHIP-NEXT: 0,
// OWNERSHIP-NEXT: 2
// DIAGONAL: diagonals=1
// DIAGONAL: non-orthogonal edge=edge_diagonal
// FOREIGN-TILE: foreign_tile_crossings=1
// FOREIGN-TILE: foreign-tile-crossing edge=edge_foreign
// NODE: node_intersections=1
// NODE: node-intersection edge=edge_node
// OVERLAP: collinear_overlaps=1 narrow_gaps=0 crossings=0
// OVERLAP: edge-separation first=edge_first second=edge_overlap overlap=40.00 gap=0.00
// NARROW: collinear_overlaps=0 narrow_gaps=1 crossings=0
// NARROW: edge-separation first=edge_first second=edge_narrow overlap=40.00 gap=0.80
// CROSSING: collinear_overlaps=0 narrow_gaps=0 crossings=1
// CROSSING: edge-crossing first=edge_horizontal second=edge_vertical at=(50.0, 30.0)
// JSON-OWNERSHIP: foreign_tile_crossings=2
// JSON-OWNERSHIP-NOT: foreign-tile-crossing edge=edge_defect
// JSON-OWNERSHIP: foreign-tile-crossing edge=edge_narrow

module {
  aie.device(npu1_1col) {
    func.func private @kernel(memref<16xi32>)

    %tile = aie.tile(0, 2) loc("geometry.py":10:1)
    %sink = aie.tile(0, 3) loc("geometry.py":11:1)
    %buffer = aie.buffer(%tile) {sym_name = "data"} : memref<16xi32> loc("geometry.py":12:1)
    aie.core(%tile) {
      func.call @kernel(%buffer) : (memref<16xi32>) -> () loc("geometry.py":14:1)
      aie.end
    }
    aie.mem(%tile) {
      %0 = aie.dma_start(MM2S, 0, ^bd, ^end)
    ^bd:
      aie.dma_bd(%buffer : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^end
    ^end:
      aie.end
    }
    aie.flow(%tile, DMA : 0, %sink, Core : 0) loc("geometry.py":25:1)
  }
}