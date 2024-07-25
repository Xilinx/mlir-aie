//===- test_mmap0.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-mmap %s | FileCheck %s
// RUN: aie-translate --tilecol=4 --tilerow=4 --aie-generate-bcf %s | FileCheck --check-prefix=BCF44 %s
// RUN: aie-translate --tilecol=4 --tilerow=4 --aie-generate-ldscript %s | FileCheck --check-prefix=LD44 %s

// CHECK-LABEL: Tile(3, 4)
// CHECK: _symbol x 0x28000 32
// CHECK: _symbol a 0x38000 16
// CHECK: _symbol b 0x38010 64
// CHECK: _symbol c 0x38050 1024
// CHECK-LABEL: Tile(4, 3)
// CHECK: _symbol a 0x30000 16
// CHECK: _symbol b 0x30010 64
// CHECK: _symbol c 0x30050 1024
// CHECK: _symbol z 0x38000 32
// CHECK-LABEL: Tile(4, 4)
// CHECK: _symbol z 0x20000 32
// CHECK: _symbol a 0x28000 16
// CHECK: _symbol b 0x28010 64
// CHECK: _symbol c 0x28050 1024
// CHECK: _symbol t 0x30000 32
// CHECK: _symbol y 0x38000 32
// CHECK-LABEL: Tile(4, 5)
// CHECK: _symbol a 0x20000 16
// CHECK: _symbol b 0x20010 64
// CHECK: _symbol c 0x20050 1024
// CHECK: _symbol t 0x38000 32
// CHECK-LABEL: Tile(5, 4)
// CHECK: _symbol y 0x28000 32

// BCF44:      _entry_point _main_init
// BCF44-NEXT: _symbol core_4_4 _after _main_init
// BCF44-NEXT: _symbol _main_init 0
// BCF44-NEXT: _reserved DMb 0x00000 0x20000 // Don't put data in code memory
// BCF44-NEXT: _stack DM_stack 0x28000 0x400 // stack for core
// BCF44:      // mapping neighbors tile memory
// BCF44-NEXT: // south -------------------------------------------------
// BCF44-NEXT: _reserved DMb 0x20000 0x8000  // Don't allocate variables in south neighbor
// BCF44:      _symbol z 0x20000 32
// BCF44-NEXT: _extern z
// BCF44-NEXT: _reserved DMb 0x20000 32
// BCF44:      // west -------------------------------------------------
// BCF44-NEXT: _symbol a 0x28000 16
// BCF44-NEXT: _extern a
// BCF44-NEXT: _reserved DMb 0x28000 16
// BCF44:      _symbol b 0x28010 64
// BCF44-NEXT: _extern b
// BCF44-NEXT: _reserved DMb 0x28010 64
// BCF44:      _symbol c 0x28050 1024
// BCF44-NEXT: _extern c
// BCF44-NEXT: _reserved DMb 0x28050 1024
// BCF44:      // north -------------------------------------------------
// BCF44-NEXT: _reserved DMb 0x30000 0x8000  // Don't allocate variables in north neighbor
// BCF44:      _symbol t 0x30000 32
// BCF44-NEXT: _extern t
// BCF44-NEXT: _reserved DMb 0x30000 32
// BCF44:       // east -------------------------------------------------
// BCF44-NEXT: _reserved DMb 0x38000 0x8000  // Don't allocate variables in east neighbor
// BCF44:      _symbol y 0x38000 32
// BCF44-NEXT: _extern y
// BCF44-NEXT: _reserved DMb 0x38000 32
// BCF44:      // end mapping neighbors tile memory
// BCF44:      _reserved DMb 0x40000 0xC0000 // And everything else the core can't see
// BCF44-NEXT: _resolve _main core_4_4


// LD44: MEMORY
// LD44-NEXT: {
// LD44-NEXT:    program (RX) : ORIGIN = 0, LENGTH = 0x0020000
// LD44-NEXT:    data (!RX) : ORIGIN = 0x28450, LENGTH = 0x7BB0
// LD44-NEXT: }
// LD44-NEXT: ENTRY(_main_init)
// LD44-NEXT: SECTIONS
// LD44-NEXT: {
// LD44-NEXT:   . = 0x0;
// LD44-NEXT:  .text : {
// LD44-NEXT:     /* the _main_init symbol has to come at address zero. */
// LD44-NEXT:     *crt0.o(.text)
// LD44-NEXT:     . = 0x200;
// LD44-NEXT:     _ctors_start = .;
// LD44-NEXT:     _init_array_start = .;
// LD44-NEXT:     KEEP(SORT(*.init_array))
// LD44-NEXT:     _ctors_end = .;
// LD44-NEXT:     _init_array_end = .;
// LD44-NEXT:     _dtors_start = .;
// LD44-NEXT:     _dtors_end = .;
// LD44-NEXT:     *(.text)
// LD44-NEXT:  } > program
// LD44-NEXT:  .data : {
// LD44-NEXT:     *(.data*);
// LD44-NEXT:     *(.rodata*)
// LD44-NEXT:  } > data
// LD44-NEXT:   . = 0x28000;
// LD44-NEXT:   _sp_start_value_DM_stack = .;
// LD44-NEXT:   . += 0x400;
// LD44-NEXT: . = 0x20000
// LD44-NEXT: z = .;
// LD44-NEXT: . += 0x20
// LD44-NEXT: . = 0x28000
// LD44-NEXT: a = .;
// LD44-NEXT: . += 0x10
// LD44-NEXT: . = 0x28010
// LD44-NEXT: b = .;
// LD44-NEXT: . += 0x40
// LD44-NEXT: . = 0x28050
// LD44-NEXT: c = .;
// LD44-NEXT: . += 0x400
// LD44-NEXT: . = 0x30000
// LD44-NEXT: t = .;
// LD44-NEXT: . += 0x20
// LD44-NEXT: . = 0x38000
// LD44-NEXT: y = .;
// LD44-NEXT: . += 0x20

module @test_mmap0 {
 aie.device(xcvc1902) {
  %t44 = aie.tile(4, 4)
  %t34 = aie.tile(3, 4)
  %t54 = aie.tile(5, 4)
  %t43 = aie.tile(4, 3)
  %t45 = aie.tile(4, 5)

  %buf44_0 = aie.buffer(%t44) { sym_name = "a", address = 0x0 : i32 } : memref<4xi32>
  %buf44_1 = aie.buffer(%t44) { sym_name = "b", address = 0x10 : i32  } : memref<16xi32>
  %buf44_2 = aie.buffer(%t44) { sym_name = "c", address = 0x50 : i32 } : memref<256xi32>
  %buf34_0 = aie.buffer(%t34) { sym_name = "x", address = 0x0 : i32 } : memref<8xi32>
  %buf54_0 = aie.buffer(%t54) { sym_name = "y", address = 0x0 : i32 } : memref<8xi32>
  %buf43_0 = aie.buffer(%t43) { sym_name = "z", address = 0x0 : i32 } : memref<8xi32>
  %buf45_0 = aie.buffer(%t45) { sym_name = "t", address = 0x0 : i32 } : memref<8xi32>

  aie.core(%t44) {
    aie.end
  }
 }
}

