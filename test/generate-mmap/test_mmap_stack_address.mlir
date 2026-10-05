// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Each bank region excludes buffers, the recorded data region, and the moved
// stack. Both linkers use the same tile-relative stack address.
// RUN: aie-translate --tilecol=0 --tilerow=2 --aie-generate-ldscript %s | FileCheck %s --check-prefix=LD
// RUN: aie-translate --tilecol=0 --tilerow=2 --aie-generate-bcf %s | FileCheck %s --check-prefix=BCF

// LD: data (!RX) : ORIGIN = 0x78000, LENGTH = 0x1000
// LD-NEXT: bank0 (!RX) : ORIGIN = 0x70000, LENGTH = 0x2000
// LD-NEXT: bank1 (!RX) : ORIGIN = 0x74400, LENGTH = 0x3C00
// LD-NEXT: bank2 (!RX) : ORIGIN = 0x79000, LENGTH = 0x3000
// LD-NEXT: bank3 (!RX) : ORIGIN = 0x7C000, LENGTH = 0x4000
// LD: . = 0x74000;
// LD-NEXT: _sp_start_value_DM_stack = .;
// LD-NEXT: . += 0x400; /* stack */
// BCF: _stack DM_stack 0x74000 0x400 // stack for core

module {
  aie.device(npu1_1col) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) { sym_name = "b", address = 8192 : i32 } : memref<8192xi8>
    %data = aie.buffer(%t) { sym_name = "core_data", address = 32768 : i32, core_data } : memref<4096xi8>
    %c = aie.core(%t) { aie.end } { stack_size = 1024 : i32, stack_address = 16384 : i32, stack_bank = 1 : i32 }
  }
}
