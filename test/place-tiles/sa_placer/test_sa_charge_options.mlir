//===- test_sa_charge_options.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-place-tiles="placer=sa_placer sa-walked-fifos=x" %s 2>&1 | FileCheck %s --check-prefix=WALKED
// RUN: not aie-opt --aie-place-tiles="placer=sa_placer sa-core-data-sizes=@d0:x=1" %s 2>&1 | FileCheck %s --check-prefix=SIZES
// RUN: not aie-opt --aie-place-tiles="placer=sa_placer sa-walked-fifos=@dx::@x" %s 2>&1 | FileCheck %s --check-prefix=DEVICE
// RUN: not aie-opt --aie-place-tiles="placer=sa_placer sa-walked-fifos=@d1::@y" %s 2>&1 | FileCheck %s --check-prefix=FIFO
// RUN: not aie-opt --aie-place-tiles="placer=sa_placer sa-core-data-sizes=@d1:1=8" %s 2>&1 | FileCheck %s --check-prefix=CORE
// RUN: aie-opt --aie-place-tiles="placer=sa_placer sa-walked-fifos=@d1::@x sa-core-data-sizes=@d0:0=8" %s -o /dev/null

// Each charge names a device, a fifo or core in it, and bytes; a malformed
// entry or one naming another device is reported once, not once a device,
// and one naming something the device does not have is an error, not a
// charge that silently does nothing.
// WALKED: error: sa-walked-fifos entry 'x' is not @<device>::@<fifo>
// WALKED-NOT: error:
// SIZES: error: sa-core-data-sizes entry '@d0:x=1' is not @<device>:<core>=<bytes>[@<align>]
// SIZES-NOT: error:
// DEVICE: error: SA placer charges name no device @dx
// DEVICE-NOT: error:
// FIFO: error: sa-walked-fifos names no objectFIFO @y in @d1
// CORE: error: sa-core-data-sizes names core 1 of @d1, which has 1 cores

module {
  aie.device(npu2) @d0 {
    %c = aie.logical_tile<CoreTile>(?, ?)
    %m = aie.logical_tile<MemTile>(?, ?)
    aie.objectfifo @x(%m, {%c}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @y(%c, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%c) {
      aie.end
    }
  }
  aie.device(npu2) @d1 {
    %c = aie.logical_tile<CoreTile>(?, ?)
    %m = aie.logical_tile<MemTile>(?, ?)
    aie.objectfifo @x(%m, {%c}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%c) {
      aie.end
    }
  }
}
