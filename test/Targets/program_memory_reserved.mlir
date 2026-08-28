//===- program_memory_reserved.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A core that reserves program memory gets a correspondingly shorter program
// region, so its own code growing into the reservation is an ordinary link
// error naming the section and the overrun, rather than a silent overwrite of
// the running program.
//
// The reservation exists for code written at run time -- program-memory
// overlays. Without it that boundary can only be enforced by an ASSERT smuggled
// into the core's link through a link_with fragment, which works only because
// ld.lld parses an unrecognised INPUT() as a linker script.
//
// 0x4000 of program memory less an 0x2000 reservation leaves 0x2000 for code.

// RUN: aie-translate --aie-generate-ldscript --tilecol=0 --tilerow=2 %s | FileCheck %s
// CHECK: program (RX) : ORIGIN = 0, LENGTH = 0x2000

module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %c = aie.core(%t) {
      aie.end
    } { program_memory_reserved = 8192 : i32 }
  }
}
