//===- reconfig_method_ctrlpkt_init_from_overlay.mlir ------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: peano

// --reconfig-method=ctrlpkt synthesizes ONE shared `main:init` that stands up
// the resident @ctrl_pkt_overlay. `init` must be DESIGN-INDEPENDENT: its
// signature is only the uniform trailing ctrl-pkt-stream buffer that
// AIECtrlPacketToDma appends to every runtime sequence (memref<?xi32>), and
// its body is nothing but the overlay's own `load_pdi`. It must NOT inherit
// the first folded design's I/O arg (here reconfig_twodevice_a.mlir's
// memref<4xi32>) or the dead arith.constant preamble that a naive
// clone-and-truncate of that design's sequence would carry over. Same
// two-config fixture as reconfig_method_write32_no_overlay.mlir, guarding
// npu_lowered.mlir (the post-split module) end to end.

// RUN: rm -rf %t && mkdir -p %t
// RUN: cd %t && aiecc --get-full-elf --reconfig-method=ctrlpkt --get-input-with-addresses --get npu_lowered.mlir --tmpdir=%t %S/Inputs/reconfig_twodevice_a.mlir %S/Inputs/reconfig_twodevice_b.mlir 2>&1
// RUN: cat %t/npu_lowered.mlir | FileCheck %s

// `init`'s signature is exactly one block arg: the trailing memref<?xi32>
// ctrl-pkt buffer. Design A's memref<4xi32> arg must be gone.
// CHECK: aie.runtime_sequence @init(%{{[a-zA-Z0-9_]+}}: memref<?xi32>
// CHECK-NOT: aie.runtime_sequence @init(%{{[a-zA-Z0-9_]+}}: memref<4xi32>

// `init`'s body is the overlay's load_pdi and nothing else: no dead
// arith.constant preamble, no other op before the closing brace.
// CHECK-NEXT: aiex.npu.load_pdi {device_ref = @ctrl_pkt_overlay, expand_mode = 0 : i32, id = {{[0-9]+}} : i32}
// CHECK-NEXT: }

// The two config devices are still folded in (sanity check: the build did
// not silently drop a config while fixing init's shape).
// CHECK: aie.device{{.*}}@cfg_a
// CHECK: aie.device{{.*}}@cfg_b
