//===- reconfig_autopacketize_default.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: peano

// auto-packetize-control-ingress is on by default in the resident-overlay
// flow now (no --ctrlpkt-auto-packetize needed). The fixture
// (reconfig_idiomatic_twochannel.mlir) saturates both of column 0's shim
// MM2S channels with circuit-switched ingress, which would otherwise leave
// no free channel for the control overlay.

// Case 1 (default, no flag at all): the pass still runs, promotes @in1 to
// packet mode (warning, not remark), and the overlay routes successfully.
// RUN: aiecc --get-full-elf --reconfig-method=ctrlpkt --get-input-with-addresses --tmpdir=%t_default --verbose %S/Inputs/reconfig_idiomatic_twochannel.mlir 2>&1 | FileCheck %s --check-prefix=DEFAULT
// DEFAULT: warning: auto-packetized objectFifo 'in1' on column 0 from circuit -> packet for resident control coexistence
// DEFAULT: wrote edge 'input_with_addresses.mlir'

// Case 2 (legacy opt-in flag, back-compat): the old --ctrlpkt-auto-packetize
// flag is still accepted (e.g. the corpus sweep harness passes it explicitly)
// and does not double-apply the pass -- exactly one flip warning, same as
// the default.
// RUN: aiecc --get-full-elf --reconfig-method=ctrlpkt --ctrlpkt-auto-packetize --get-input-with-addresses --tmpdir=%t_legacy --verbose %S/Inputs/reconfig_idiomatic_twochannel.mlir 2>&1 | FileCheck %s --check-prefix=LEGACY
// LEGACY-COUNT-1: warning: auto-packetized objectFifo 'in1' on column 0
// LEGACY-NOT: warning: auto-packetized objectFifo 'in1' on column 0
// LEGACY: wrote edge 'input_with_addresses.mlir'

// Case 3 (opt-out): --ctrlpkt-auto-packetize=false disables the default
// and reproduces the pre-existing shim-MM2S-exhaustion wall.
// RUN: not aiecc --get-full-elf --reconfig-method=ctrlpkt --ctrlpkt-auto-packetize=false --get-input-with-addresses --tmpdir=%t_optout %S/Inputs/reconfig_idiomatic_twochannel.mlir 2>&1 | FileCheck %s --check-prefix=OPTOUT
// OPTOUT: failed to generate column control overlay: all shim mm2s dma channels for column 0 are reserved by circuit-switched flows
