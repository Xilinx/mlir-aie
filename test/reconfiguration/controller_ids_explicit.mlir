// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s --check-prefix=PLAIN
// RUN: aie-opt --aie-objectFifo-stateful-transform="reserve-controller-packet-ids=true" %s | FileCheck %s --check-prefix=RESERVED

// PLAIN: aie.packet_flow(0)
// RESERVED-NOT: aie.packet_flow(0)
// RESERVED: aie.packet_flow(1)
module {
  aie.device(npu2) @explicit_controller {
    %shim = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
    %core = aie.tile(0, 2)
    aie.objectfifo @input(%shim, {%core}, 1 : i32) {transport = #aie.transport<auto, packet = #aie.packet_info<>>} : !aie.objectfifo<memref<4xi32>>
    aie.core(%core) {
      %a = aie.objectfifo.acquire @input(Consume, 1) : memref<4xi32>
      aie.objectfifo.release @input(Consume, 1)
      aie.end
    }
  }
}
