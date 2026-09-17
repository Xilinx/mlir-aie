//===- large_stack.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A 20 KiB stack spans two 16 KiB banks. Buffers must start above the
// reservation, and both linker formats must preserve its full size.
// large_stack_peano.test and large_stack_chess.test compile this same design.

// RUN: aie-opt --aie-assign-buffer-addresses="alloc-scheme=bank-aware" %s -o %t.mlir
// RUN: FileCheck %s --check-prefix=ADDR < %t.mlir
// RUN: aie-translate --aie-generate-ldscript --tilecol=0 --tilerow=2 %t.mlir | FileCheck %s --check-prefix=LD
// RUN: aie-translate --aie-generate-bcf --tilecol=0 --tilerow=2 %t.mlir | FileCheck %s --check-prefix=BCF

// ADDR: aie.buffer(%tile_0_2) {address = 20480 : i32, mem_bank = 1 : i32, sym_name = "out"} : memref<256xi32>
// ADDR: stack_size = 20480 : i32

// LD: . = 0x70000;
// LD-NEXT: _sp_start_value_DM_stack = .;
// LD-NEXT: . += 0x5000; /* stack */
// LD: . = 0x75000;
// LD-NEXT: out = .;

// BCF: _stack DM_stack 0x70000 0x5000
// BCF: _symbol out 0x75000 1024
// BCF: _reserved DMb 0x75000 1024

module {
  aie.device(npu1_1col) {
    %tile = aie.tile(0, 2)
    %out = aie.buffer(%tile) {sym_name = "out"} : memref<256xi32>
    func.func private @large_stack(memref<256xi32>) attributes {link_with = "large_stack_kernel.o"}
    aie.core(%tile) {
      func.call @large_stack(%out) : (memref<256xi32>) -> ()
      aie.end
    } {stack_size = 20480 : i32}
  }
}
