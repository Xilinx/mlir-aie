// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-objectfifo-split --aie-assign-packet-ids="packet-sw-objFifos=true" %s | FileCheck %s

// A packet rule already in the device matches ids 0 to 3, so the
// packet-switched objectFIFO takes id 4.
// CHECK-LABEL: @rules
// CHECK: packet = #aie.packet_info<pkt_id = 4>

module {
  aie.device(npu2) @rules {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    %far = aie.tile(3, 2)
    aie.switchbox(%far) {
      %a = aie.amsel<0> (0)
      aie.masterset(DMA : 0, %a)
      aie.packet_rules(South : 0) {
        aie.rule(28, 0, %a)
      }
    }
    aie.objectfifo @a(%shim, {%core}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%core) {
      %a = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a(Consume, 1)
      aie.end
    }
  }
}
