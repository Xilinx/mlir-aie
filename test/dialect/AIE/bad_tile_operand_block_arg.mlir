//===- bad_tile_operand_block_arg.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// A tile operand is an index, so a block argument parses where an aie.tile
// goes. It has no defining op, and the verifiers that look at the operand's
// defining op reject it instead of dereferencing the missing op. The generic
// form is the way to write a device with a block argument.

// The producer or a consumer of an objectfifo.
"aie.device"() ({
^bb0(%tile: index):
  %t = aie.tile(1, 3)
  // expected-error @+1 {{producer tile must implement TileLike interface}}
  aie.objectfifo @of(%tile, {%t}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
  "aie.end"() : () -> ()
}) {device = 4 : i32} : () -> ()

// -----

"aie.device"() ({
^bb0(%tile: index):
  %t = aie.tile(1, 3)
  // expected-error @+1 {{consumer tile must implement TileLike interface}}
  aie.objectfifo @of(%t, {%tile}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
  "aie.end"() : () -> ()
}) {device = 4 : i32} : () -> ()

// -----

// A link verified before the objectfifos that name its shared tile.
"aie.device"() ({
^bb0(%tile: index):
  %t = aie.tile(1, 3)
  // expected-error @+1 {{shared tile must implement TileLike interface}}
  aie.objectfifo.link [@a] -> [@b] ([] [])
  aie.objectfifo @a(%t, {%tile}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
  aie.objectfifo @b(%tile, {%t}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
  "aie.end"() : () -> ()
}) {device = 4 : i32} : () -> ()

// -----

// The tile an objectfifo is delegated to.
"aie.device"() ({
^bb0(%tile: index):
  %t = aie.tile(1, 3)
  aie.objectfifo @of(%t, {%t}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
  // expected-error @+1 {{'aie.objectfifo.allocate' op delegate must be a tile}}
  aie.objectfifo.allocate @of(%tile)
  "aie.end"() : () -> ()
}) {device = 4 : i32} : () -> ()

// -----

// The tile of an objectfifo pool and of its endpoints.
"aie.device"() ({
^bb0(%tile: index):
  %t = aie.tile(1, 3)
  // expected-error @+1 {{'aie.objectfifo.pool' op tile operand is not an aie.tile or aie.logical_tile}}
  aie.objectfifo.pool @p(%tile) {depth = 2 : i32} : memref<16xi32> {
    aie.objectfifo.segment @all {offset = 0 : i32, size = 16 : i32}
  }
  "aie.end"() : () -> ()
}) {device = 4 : i32} : () -> ()

// -----

"aie.device"() ({
^bb0(%tile: index):
  %t = aie.tile(1, 3)
  aie.objectfifo.pool @p(%t) {depth = 2 : i32} : memref<16xi32> {
    aie.objectfifo.segment @all {offset = 0 : i32, size = 16 : i32}
  }
  // expected-error @+1 {{'aie.objectfifo.core_endpoint' op tile operand is not an aie.tile or aie.logical_tile}}
  aie.objectfifo.core_endpoint @end(%tile) fills @p
  "aie.end"() : () -> ()
}) {device = 4 : i32} : () -> ()

// -----

"aie.device"() ({
^bb0(%tile: index):
  %t = aie.tile(1, 3)
  aie.objectfifo.pool @p(%t) {depth = 2 : i32} : memref<16xi32> {
    aie.objectfifo.segment @all {offset = 0 : i32, size = 16 : i32}
  }
  // expected-error @+1 {{'aie.objectfifo.dma_endpoint' op tile operand is not an aie.tile or aie.logical_tile}}
  aie.objectfifo.dma_endpoint @end(%tile) drains @p
  "aie.end"() : () -> ()
}) {device = 4 : i32} : () -> ()

// -----

// The tile a route endpoint names.
"aie.device"() ({
^bb0(%tile: index):
  // expected-error @+1 {{'aie.route_endpoint' op tile operand is not an aie.tile or aie.logical_tile}}
  aie.route_endpoint @ep(%tile) DMA
  "aie.end"() : () -> ()
}) {device = 4 : i32} : () -> ()

// -----

// The ends of a cascade flow.
"aie.device"() ({
^bb0(%tile: index):
  %t = aie.tile(1, 3)
  // expected-error @+1 {{source and dest must be tile-like operations}}
  aie.cascade_flow(%tile, %t)
  "aie.end"() : () -> ()
}) {device = 4 : i32} : () -> ()

// -----

"aie.device"() ({
^bb0(%tile: index):
  %t = aie.tile(1, 3)
  // expected-error @+1 {{source and dest must be tile-like operations}}
  aie.cascade_flow(%t, %tile)
  "aie.end"() : () -> ()
}) {device = 4 : i32} : () -> ()

// -----

// The tile of a cascade configuration.
"aie.device"() ({
^bb0(%tile: index):
  // expected-error @+1 {{'aie.configure_cascade' op tile operand is not an aie.tile or aie.logical_tile}}
  aie.configure_cascade(%tile, North, South)
  "aie.end"() : () -> ()
}) {device = 4 : i32} : () -> ()

// -----

// A flow's, or a packet flow's, source and destination.
"aie.device"() ({
^bb0(%tile: index):
  %t = aie.tile(1, 3)
  // expected-error @+1 {{'aie.flow' op tile operand is not an aie.tile or aie.logical_tile}}
  aie.flow(%tile, DMA : 0, %t, DMA : 0)
  "aie.end"() : () -> ()
}) {device = 4 : i32} : () -> ()

// -----

"aie.device"() ({
^bb0(%tile: index):
  %t = aie.tile(1, 3)
  // expected-error @+1 {{'aie.flow' op tile operand is not an aie.tile or aie.logical_tile}}
  aie.flow(%t, DMA : 0, %tile, DMA : 0)
  "aie.end"() : () -> ()
}) {device = 4 : i32} : () -> ()

// -----

"aie.device"() ({
^bb0(%tile: index):
  %t = aie.tile(1, 3)
  aie.packet_flow(0x1) {
    // expected-error @+1 {{'aie.packet_source' op tile operand is not an aie.tile or aie.logical_tile}}
    aie.packet_source<%tile, DMA : 0>
    aie.packet_dest<%t, DMA : 0>
  }
  "aie.end"() : () -> ()
}) {device = 4 : i32} : () -> ()

// -----

"aie.device"() ({
^bb0(%tile: index):
  %t = aie.tile(1, 3)
  aie.packet_flow(0x1) {
    aie.packet_source<%t, DMA : 0>
    // expected-error @+1 {{'aie.packet_dest' op tile operand is not an aie.tile or aie.logical_tile}}
    aie.packet_dest<%tile, DMA : 0>
  }
  "aie.end"() : () -> ()
}) {device = 4 : i32} : () -> ()
