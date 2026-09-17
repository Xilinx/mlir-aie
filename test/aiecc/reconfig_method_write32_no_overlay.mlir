//===- reconfig_method_write32_no_overlay.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: peano

// --reconfig-method=write32 is the out-of-band delivery variant of the resident
// overlay ELF: it must route each config through direct write32/blockwrite
// writes WITHOUT ever synthesizing the resident @ctrl_pkt_overlay device (that
// is the in-band arm's control-packet transport). This guards two graph edges
// end to end for a real two-config build (reconfig_twodevice_a.mlir +
// reconfig_twodevice_b.mlir, folded by the --get-full-elf --reconfig-method=ctrlpkt
// union machinery that --reconfig-method=write32 implies):
//   * input_with_addresses.mlir -- the column-control-overlay synthesis pass
//     runs here (inside getInputWithAddressesPipeline); this is where a stale
//     `ctrl` capture would (re-)introduce @ctrl_pkt_overlay.
//   * npu_lowered.mlir -- the post-split module (main:init + configs_1..N);
//     confirms the overlay stays absent all the way through, and that the
//     init/configs split (pass-level, untouched by this test) still holds:
//     exactly one @empty load_pdi (main:init's), zero per-config load_pdi.
// --reconfig-method=write32 ALWAYS synthesizes the shared main:init (@empty
// reset); the host loads it or not (dispatch => explicit reset; skip =>
// firmware teardown reset). This test guards that always-init invariant, which
// is now the write32 default (no flag needed).
// RUN: rm -rf %t && mkdir -p %t
// RUN: cd %t && aiecc --get-full-elf --reconfig-method=write32 --get-input-with-addresses --get npu_lowered.mlir --tmpdir=%t %S/Inputs/reconfig_twodevice_a.mlir %S/Inputs/reconfig_twodevice_b.mlir 2>&1
// RUN: cat %t/input_with_addresses.mlir | FileCheck %s --check-prefix=NOOVERLAY
// RUN: cat %t/input_with_addresses.mlir | FileCheck %s --check-prefix=DEVICES
// RUN: cat %t/npu_lowered.mlir | FileCheck %s --check-prefix=NOOVERLAY
// RUN: cat %t/npu_lowered.mlir | FileCheck %s --check-prefix=SPLIT
// RUN: cat %t/npu_lowered.mlir | FileCheck %s --check-prefix=ONEEMPTY

// No resident control overlay anywhere in either stage: no @ctrl_pkt_overlay
// device, and no has_ctrl_pkt_overlay tag marking another device as
// overlay-coexisting. (Not checked: the per-flow `is_ctrl_pkt_overlay` unit
// attribute on masterset/rule ops -- that tags the generic control-packet
// "class" used to route write32/blockwrite direct writes to non-shim tiles,
// which the no-overlay arm still legitimately uses; it is unrelated to a
// resident overlay device's existence.)
// NOOVERLAY-NOT: aie.device{{.*}}@ctrl_pkt_overlay
// NOOVERLAY-NOT: has_ctrl_pkt_overlay

// The two config devices are still folded in (sanity check: the build did not
// silently drop a config while dropping the overlay).
// DEVICES: aie.device{{.*}}@cfg_a
// DEVICES: aie.device{{.*}}@cfg_b

// main:init carries the sole @empty reset load; main:configs_1/configs_2 each
// carry the config's direct-write payload (blockwrite BD + address_patch of the
// runtime arg) followed by the self-clear teardown (maskwrite32/write32 from the
// unconditional self-clear for write32) and a closing sync, with no per-config
// load_pdi re-arm. The ordered payload -> teardown -> sync chain proves each
// config region is non-empty (a regression that emptied a config body would fail
// these), not merely load_pdi-free.
//
// write32 synthesizes init by CLONING the first entrypoint
// and truncating after its load_pdi (no ctrl-pkt lowering runs here, so there is
// no uniform trailing ctrl arg to key off of -- that is the ctrlpkt-only
// design-independent path). So init keeps design A's own memref<4xi32> I/O arg,
// NOT a memref<?xi32> ctrl buffer. A regression that mis-applied the ctrlpkt
// overlay-init synthesis to this mode would flip the arg type (and read a
// design arg as if it were the ctrl buffer), which this pins.
// SPLIT: aie.runtime_sequence @init(%{{[a-zA-Z0-9_]+}}: memref<4xi32>
// SPLIT: aiex.npu.load_pdi {device_ref = @empty
// SPLIT: aie.runtime_sequence @configs_1
// SPLIT-NOT: load_pdi
// SPLIT: aiex.npu.blockwrite
// SPLIT: aiex.npu.address_patch
// SPLIT: aiex.npu.maskwrite32
// SPLIT: aiex.npu.write32
// SPLIT: aiex.npu.sync
// SPLIT: aie.runtime_sequence @configs_2
// SPLIT-NOT: load_pdi
// SPLIT: aiex.npu.blockwrite
// SPLIT: aiex.npu.address_patch
// SPLIT: aiex.npu.maskwrite32
// SPLIT: aiex.npu.write32
// SPLIT: aiex.npu.sync
// SPLIT: aie.device{{.*}}@cfg_a
// SPLIT-NOT: load_pdi

// Exactly one load_pdi in the whole emitted module: init's single @empty reset.
// The COUNT-1 consumes that one; the trailing NOT bounds the rest of the file,
// so a regression re-emitting a per-config @empty_1 re-arm (its orphaned device
// decl sits right next to @empty_0) would trip this even though @configs_k are
// checked load_pdi-free above.
// ONEEMPTY-COUNT-1: aiex.npu.load_pdi
// ONEEMPTY-NOT: aiex.npu.load_pdi
