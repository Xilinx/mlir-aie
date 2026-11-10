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
  // On a control overlay's device, ctrl_pkt_multicast lists its control
  // multicasts, each a dictionary of its packet `id`, the `column` whose
  // control channel it leaves by, and its `tiles` (#aie.tile).
  let discardableAttrs = (ins "::mlir::UnitAttr":$is_ctrl_pkt_overlay,
      "::mlir::UnitAttr":$priority_route, "::mlir::ArrayAttr":$shareable_with,
      "::mlir::UnitAttr":$reserve_controller_packet_ids,
      "::mlir::ArrayAttr":$ctrl_pkt_multicast);
}

#endif // AIE_BASE
