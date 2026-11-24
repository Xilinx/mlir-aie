//===- AIE.td ----------------------------------------------*- tablegen -*-===//
//
// Copyright (C) 2019-2022 Xilinx, Inc.
// Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_BASE
#define AIE_BASE

include "mlir/IR/OpBase.td"

def AIE_Dialect : Dialect {
  let name = "aie";
  let cppNamespace = "::xilinx::AIE";
  let description = [{

    This is a dialect for describing netlists of AIE components in a
    Versal device.  It focuses on representing logical stream connections
    between cores and DMAs, along with the implementation of those logical
    connections in the various switch components.  In the dialect, a
    switch is referred to as `switchbox` to avoid confusion with the
    `switch` keyword in C/C++.

    `aie.shareable_with` is an array of flat symbol references. On a route
    endpoint it lists other route endpoints allowed to share its shim channel;
    both endpoints must permit the pairing. On a shim DMA allocation it lists
    route endpoints allowed to use that fixed channel. Empty lists forbid
    these shares. Explicit lists override lifetime analysis, but resource and
    packet-routing constraints still apply. Omission permits MM2S inference;
    S2MM sharing requires explicit permission because receive ordering does
    not constrain independent senders. The author must guarantee stream order
    and completion across repeated executions. On an unsplit objectFIFO, the
    list names other objectFIFOs and applies to its shim DMA endpoint; splitting
    transfers the permissions to route endpoints and rewrites the references.

    On a device, `aie.reserve_controller_packet_ids` reserves controller IDs
    when assigning packet IDs.

    `aie.generated_configure` marks the runtime sequence that
    convert-aie-to-transaction generated to hold its device's configuration.
    That sequence is named `configure` unless a user sequence already has the
    name, so consumers key on the attribute, not the name.

    An external `func.func` with `aie.no_lock_acquires` declares that its body
    and every transitive callee perform no AIE lock acquisitions. Core lock
    analysis uses this contract while collecting explicit lock operations.
    An external declaration without the attribute has unknown lock effects.
    This contract permits lock releases and other side effects.
    Analysis traverses the bodies of defined functions without requiring this
    attribute. Teardown generation warns and skips device lock draining if
    any core has an incomplete lock summary. It still emits core disables
    and DMA resets.

    `aie.ctrl_pkt_multicast` declares control multicast routes on an overlay
    `aie.device`. Each array entry contains three fields:

    - `allocation`: a flat symbol reference to an MM2S
      `aie.shim_dma_allocation` in that overlay.
    - `packet_id`: the stream packet ID for the route.
    - `tiles`: the route's destination coordinates as `#aie.tile<column, row>`.

    `aiex.control_packet` uses `multicast_id` as a zero-based index into this
    array. `aiex.set_device` selects the configuration and its overlay.
    The allocation supplies the source tile and channel. The encoder uses
    `packet_id` in the stream header. The DMA lowerer references the matching
    imported allocation by symbol in `dma_configure_task_for`.

    One allocation can carry several packet IDs with different destinations.
    Distinct allocations can use the same packet ID. The destination list
    describes the allocation-and-packet-ID pair, not the channel alone.
    The producer must list every destination once and match the fixed route.
    Destination lists may overlap. Packet optimizers use these lists to
    match multicast writes and retain ordering at shared destinations.

  }];
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  // Python wrappers in dialects/aie.py and AIEX lowerings emit arith.constant
  // and other operations, so this dialect depends on arith being available.
  let dependentDialects = ["::mlir::arith::ArithDialect"];
  // The op assembly formats still print inherent attributes in `attr-dict`;
  // opt out of upstream's strict-properties default until they migrate.
  let useStrictPropertiesInAssemblyFormat = 0;
  // Overlay-owned settings survive application configuration and teardown.
  let discardableAttrs = (ins "::mlir::UnitAttr":$is_ctrl_pkt_overlay,
      "::mlir::UnitAttr":$priority_route, "::mlir::ArrayAttr":$shareable_with,
      "::mlir::UnitAttr":$reserve_controller_packet_ids,
      "::mlir::UnitAttr":$generated_configure,
      "::mlir::UnitAttr":$no_lock_acquires,
      "::mlir::ArrayAttr":$ctrl_pkt_multicast);
}

#endif // AIE_BASE
