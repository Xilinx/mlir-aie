//===- mixed_switching.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Switching is chosen per fifo, so circuit- and packet-switched connections
// coexist. A pinned packet_id is honoured; the rest are assigned around it and
// around packet flows the design already declares.

// RUN: aie-opt --aie-objectfifo-split %s | FileCheck %s --check-prefix=SPLIT
// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" %s | FileCheck %s

module @mixed {
  aie.device(xcve2302) {
    %a = aie.tile(1, 2)
    %b = aie.tile(3, 2)
    %c = aie.tile(1, 3)
    %d = aie.tile(3, 3)

    aie.objectfifo @plain (%a, {%b}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @auto (%c, {%d}, 2 : i32) {packet} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @pinned (%b, {%c}, 2 : i32) {packet, packet_id = 7 : i8} : !aie.objectfifo<memref<16xi32>>
  }
}

// The choice rides on the flow, which is what becomes one kind of route or the
// other.
// SPLIT:       aie.route from @plain_prod_dma to [@plain_cons_dma]
// SPLIT-NOT:   packet
// SPLIT:       aie.route from @auto_prod_dma to [@auto_cons_dma] {packet}
// SPLIT:       aie.route from @pinned_prod_dma to [@pinned_cons_dma] {packet, packet_id = 7 : i8}

// CHECK-DAG:   %[[A:.*]] = aie.tile(1, 2)
// CHECK-DAG:   %[[B:.*]] = aie.tile(3, 2)
// CHECK-DAG:   %[[C:.*]] = aie.tile(1, 3)
// CHECK-DAG:   %[[D:.*]] = aie.tile(3, 3)

// CHECK:       aie.flow(%[[A]], DMA : 0, %[[B]], DMA : 0)
// CHECK:       aie.packet_flow(0) {
// CHECK:         aie.packet_source<%[[C]], DMA : 0>
// CHECK:         aie.packet_dest<%[[D]], DMA : 0>
// CHECK:       }
// CHECK:       aie.packet_flow(7) {
// CHECK:         aie.packet_source<%[[B]], DMA : 0>
// CHECK:         aie.packet_dest<%[[C]], DMA : 0>
// CHECK:       }

// Only the packet-switched sources stamp a header.
// CHECK:       aie.mem(%[[A]])
// CHECK-NOT:     aie.dma_bd_packet
// CHECK:       aie.mem(%[[B]])
// CHECK:         aie.dma_bd_packet(0, 7)
