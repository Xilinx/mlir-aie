//===- write32-reset-free.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Tests that reset-free is reset-free by default: with-reset
// defaults to false, so no @empty init preload load_pdi is emitted at all --
// the firmware resets the partition on context teardown, so the preload is
// redundant. with-reset=true restores the reset preload (load_pdi @empty).

// RUN: aie-opt --aie-expand-load-pdi="reset-free=true self-clear=true" %s | FileCheck %s
// RUN: aie-opt --aie-expand-load-pdi="reset-free=true with-reset=true self-clear=true" %s | FileCheck %s --check-prefix=WITHRESET

// CHECK-NOT: load_pdi
// CHECK-NOT: @empty

module {
  aie.device(npu2_1col) @cfg {
    %shim = aie.tile(0, 0)
    %compute = aie.tile(0, 2)
    %buf = aie.buffer(%compute) {address = 1024 : i32, sym_name = "buf"} : memref<16xi32>
    %mem = aie.mem(%compute) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.next_bd ^bb1
    ^bb2:
      aie.end
    }
  }

  aie.device(npu2_1col) @main {
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      // Default (reset-free): no preload load_pdi at all -- the config
      // lowers straight to direct write32s/blockwrite (Transaction output),
      // same as the DMA BD + MM2S_0 channel-enable write seen in
      // write32-no-overlay.mlir.
      // CHECK: aiex.npu.blockwrite
      // CHECK-DAG: %[[V0:.*]] = arith.constant 0 : i32
      // CHECK-DAG: %[[A0:.*]] = arith.constant 2219540 : i32
      // CHECK: aiex.npu.write32(%[[A0]], %[[V0]]) : i32, i32
      // CHECK-DAG: %[[V1:.*]] = arith.constant 1 : i32
      // CHECK-DAG: %[[A1:.*]] = arith.constant 2219536 : i32
      // CHECK: aiex.npu.write32(%[[A1]], %[[V1]]) : i32, i32
      // Self-clear's DMA teardown still runs (the switch teardown is empty --
      // there is no switchbox in this fixture): MM2S_0 (2219536) reset
      // assert/deassert (value=2/0, mask=2).
      // CHECK-DAG: %[[A2:.*]] = arith.constant 2219536 : i32
      // CHECK: aiex.npu.maskwrite32(%[[A2]], %{{.*}}, %{{.*}}) : i32, i32, i32
      // CHECK-DAG: %[[V3:.*]] = arith.constant 0 : i32
      // CHECK-DAG: %[[M3:.*]] = arith.constant 2 : i32
      // CHECK-DAG: %[[A3:.*]] = arith.constant 2219536 : i32
      // CHECK: aiex.npu.maskwrite32(%[[A3]], %[[V3]], %[[M3]]) : i32, i32, i32

      // WITHRESET: aiex.npu.load_pdi {device_ref = @empty_0, expand_mode = 0 : i32}
      // WITHRESET: aiex.npu.blockwrite
      aiex.npu.load_pdi { device_ref = @cfg }
    }
  }
}
