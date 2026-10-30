// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: rm -rf %t && mkdir -p %t/adaptive %t/fixed
// RUN: cd %t/adaptive && %aiecc --load-pdi-to-ctrl-pkt --get=input_physical.mlir %S/../npu-xrt/reconfiguration/auto_packetize/aie.mlir
// RUN: FileCheck %s < %t/adaptive/input_physical.mlir
// RUN: cd %t/fixed && %aiecc --load-pdi-to-ctrl-pkt --ctrlpkt-adaptive-overlay=false --get=input_physical.mlir %S/../npu-xrt/reconfiguration/auto_packetize/aie.mlir
// RUN: FileCheck %s < %t/fixed/input_physical.mlir

// CHECK-LABEL: aie.device(npu2) @cfg_add
// CHECK-NOT: aie.memtile_dma
// CHECK: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0(%{{.*}}, MM2S, 0)
// CHECK-SAME: aie.fixed
// CHECK: aie.shim_dma_allocation @a_shim_alloc(%{{.*}}, MM2S, 0, <pkt_id = {{[0-9]+}}>)
// CHECK: aie.shim_dma_allocation @b_shim_alloc(%{{.*}}, MM2S, 1)
// CHECK-NOT: aie.memtile_dma
// CHECK-LABEL: aie.device(npu2) @cfg_sub
// CHECK: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0(%{{.*}}, MM2S, 0)
// CHECK-SAME: aie.fixed
// CHECK: aie.shim_dma_allocation @a_shim_alloc(%{{.*}}, MM2S, 0, <pkt_id = {{[0-9]+}}>)
// CHECK: aie.shim_dma_allocation @b_shim_alloc(%{{.*}}, MM2S, 1)
// CHECK-LABEL: aie.device(npu2) @ctrl_pkt_overlay
// CHECK: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0(%{{.*}}, MM2S, 0)
// CHECK-SAME: aie.fixed
