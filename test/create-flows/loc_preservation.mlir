//===- loc_preservation.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Connections inherit the flow location. Switchboxes, shim muxes and wires
// inherit their tile's location; tiles synthesized along the route use the
// device location. Generic printing also exposes synthesized terminators.

// RUN: aie-opt --aie-create-pathfinder-flows --mlir-print-debuginfo %s | FileCheck %s --implicit-check-not='loc(unknown)'
// RUN: aie-opt --aie-create-pathfinder-flows --mlir-print-debuginfo --mlir-print-op-generic %s | FileCheck %s --check-prefix=GENERIC --implicit-check-not='loc(unknown)'

#flow_loc = loc("user_design.py":42:4)
#tile_loc = loc("user_design.py":50:8)
#shim_loc = loc("user_design.py":51:8)
#device_loc = loc("user_design.py":5:0)

module {
  aie.device(xcvc1902) {
    %0 = aie.tile(2, 3) loc(#tile_loc)
    %1 = aie.tile(2, 0) loc(#shim_loc)
    aie.flow(%0, Core : 1, %1, DMA : 1) loc(#flow_loc)
  } loc(#device_loc)
} loc("user_design.py":1:0)

// CHECK: %tile_2_3 = aie.tile(2, 3) loc(#[[TILELOC:loc[0-9]*]])
// CHECK: %[[SHIMTILE:.*]] = aie.tile(2, 0) loc(#[[SHIMLOC:loc[0-9]*]])
// CHECK: %switchbox_2_0 = aie.switchbox(%[[SHIMTILE]])
// CHECK-NEXT: aie.connect<{{.*}}> loc(#[[FLOWLOC:loc[0-9]*]])
// CHECK-NEXT: } loc(#[[SHIMLOC]])
// CHECK: aie.shim_mux(%[[SHIMTILE]])
// CHECK-NEXT: aie.connect<North : 3, DMA : 1> loc(#[[FLOWLOC]])
// CHECK-NEXT: } loc(#[[SHIMLOC]])
// CHECK: %tile_2_1 = aie.tile(2, 1) loc(#[[DEVICELOC:loc[0-9]*]])
// CHECK: %switchbox_2_1 = aie.switchbox(%tile_2_1)
// CHECK-NEXT: aie.connect<{{.*}}> loc(#[[FLOWLOC]])
// CHECK-NEXT: } loc(#[[DEVICELOC]])
// CHECK: %tile_2_2 = aie.tile(2, 2) loc(#[[DEVICELOC]])
// CHECK: %switchbox_2_2 = aie.switchbox(%tile_2_2)
// CHECK-NEXT: aie.connect<{{.*}}> loc(#[[FLOWLOC]])
// CHECK-NEXT: } loc(#[[DEVICELOC]])
// CHECK: %switchbox_2_3 = aie.switchbox(%tile_2_3)
// CHECK-NEXT: aie.connect<{{.*}}> loc(#[[FLOWLOC]])
// CHECK-NEXT: } loc(#[[TILELOC]])
// CHECK-DAG: aie.wire(%tile_2_3 : Core, %switchbox_2_3 : Core) loc(#[[TILELOC]])
// CHECK-DAG: aie.wire(%tile_2_1 : Core, %switchbox_2_1 : Core) loc(#[[DEVICELOC]])
// CHECK-DAG: #[[FLOWLOC]] = loc("user_design.py":42:4)
// CHECK-DAG: #[[TILELOC]] = loc("user_design.py":50:8)
// CHECK-DAG: #[[SHIMLOC]] = loc("user_design.py":51:8)
// CHECK-DAG: #[[DEVICELOC]] = loc("user_design.py":5:0)

// GENERIC: "aie.switchbox"
// GENERIC: "aie.end"() : () -> () loc(#[[SHIMLOC:loc[0-9]*]])
// GENERIC: "aie.shim_mux"
// GENERIC: "aie.end"() : () -> () loc(#[[SHIMLOC]])
// GENERIC: "aie.switchbox"
// GENERIC: "aie.end"() : () -> () loc(#[[DEVICELOC:loc[0-9]*]])
// GENERIC: "aie.switchbox"
// GENERIC: "aie.end"() : () -> () loc(#[[DEVICELOC]])
// GENERIC: "aie.switchbox"
// GENERIC: "aie.end"() : () -> () loc(#[[TILELOC:loc[0-9]*]])
// GENERIC-DAG: #[[SHIMLOC]] = loc("user_design.py":51:8)
// GENERIC-DAG: #[[DEVICELOC]] = loc("user_design.py":5:0)
// GENERIC-DAG: #[[TILELOC]] = loc("user_design.py":50:8)
