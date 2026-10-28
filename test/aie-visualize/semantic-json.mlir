//===- semantic-json.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: aie-visualize
// RUN: aie-visualize --emit-json --contract=port %s | FileCheck %s --check-prefix=JSON
// RUN: aie-visualize --emit-dot --contract=port %s | FileCheck %s --check-prefix=DOT
// RUN: not aie-visualize --emit-dot --emit-json %s 2>&1 | FileCheck %s --check-prefix=FORMAT

// JSON: "id": "edge__tile_0_2__tile_0_3"
// JSON: "routes": [
// JSON-NEXT: 0
// JSON: "source": "tile_0_2"
// JSON: "sourceLocations": [
// JSON: "column": 8
// JSON: "file": "design.py"
// JSON: "line": 30
// JSON: "target": "tile_0_3"
// JSON: "format": "aie-semantic-graph"
// JSON: "id": "tile_0_2"
// JSON: "kind": "tile"
// JSON: "label": "(0,2)"
// JSON: "column": 4
// JSON: "file": "design.py"
// JSON: "line": 10
// JSON: "version": 1

// DOT: tile_0_2 [shape=box, id="tile_0_2"
// DOT-SAME: class="semantic-node source-location-10-4"
// DOT: tile_0_2 -> tile_0_3
// DOT-SAME: id="edge__tile_0_2__tile_0_3"
// DOT-SAME: source-location-30-8

// FORMAT: --emit-dot, --emit-json, and --physical are mutually exclusive

module {
  aie.device(npu1_1col) {
    %source = aie.tile(0, 2) loc("design.py":10:4)
    %dest = aie.tile(0, 3) loc("design.py":20:6)
    aie.flow(%source, Core : 0, %dest, Core : 0) loc("design.py":30:8)
  }
}
