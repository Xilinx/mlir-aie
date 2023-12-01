//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate %s --aie-generate-airbin --airbin-output-filepath=%T/airbin.elf --airbin-aux-core-dir-path=%S && %LLVM_TOOLS_DIR/obj2yaml %T/airbin.elf | FileCheck %s

// CHECK: --- !ELF
// CHECK: FileHeader:
// CHECK:   Class:           ELFCLASS64
// CHECK:   Data:            ELFDATA2LSB
// CHECK:   OSABI:           ELFOSABI_GNU
// CHECK:   Type:            ET_NONE
// CHECK:   Machine:         0xE1
// CHECK: Sections:
// CHECK:   - Name:            .sdma.bd
// CHECK:     Type:            SHT_PROGBITS
// CHECK:     Flags:           [ SHF_ALLOC ]
// CHECK:     Address:         0x301D000
// CHECK:     AddressAlign:    0x1
// CHECK:     Content:         '00000000'
// CHECK:   - Name:            .ssmast
// CHECK:     Type:            SHT_PROGBITS
// CHECK:     Flags:           [ SHF_ALLOC ]
// CHECK:     Address:         0x303F000
// CHECK:     AddressAlign:    0x1
// CHECK:     Content:         '00000000'
// CHECK:   - Name:            .ssslve
// CHECK:     Type:            SHT_PROGBITS
// CHECK:     Flags:           [ SHF_ALLOC ]
// CHECK:     Address:         0x303F100
// CHECK:     AddressAlign:    0x1
// CHECK:     Content:         '00000000'
// CHECK:   - Name:            .sspckt
// CHECK:     Type:            SHT_PROGBITS
// CHECK:     Flags:           [ SHF_ALLOC ]
// CHECK:     Address:         0x303F200
// CHECK:     AddressAlign:    0x1
// CHECK:     Content:         '00000000'
// CHECK:   - Name:            .data.mem
// CHECK:     Type:            SHT_PROGBITS
// CHECK:     Flags:           [ SHF_ALLOC ]
// CHECK:     Address:         0x3040000
// CHECK:     AddressAlign:    0x1
// CHECK:     Content:         '00000000'
// CHECK:   - Name:            '.sdma.bd (1)'
// CHECK:     Type:            SHT_PROGBITS
// CHECK:     Flags:           [ SHF_ALLOC ]
// CHECK:     Address:         0x305D000
// CHECK:     AddressAlign:    0x1
// CHECK:     Content:         '00000000'
// CHECK:   - Name:            .tdma.ctl
// CHECK:     Type:            SHT_PROGBITS
// CHECK:     Flags:           [ SHF_ALLOC ]
// CHECK:     Address:         0x305DE00
// CHECK:     AddressAlign:    0x1
// CHECK:     Content:         '00000000'
// CHECK:   - Type:            SHT_PROGBITS
// CHECK:     Flags:           [ SHF_ALLOC ]
// CHECK:     Address:         0x305DE10
// CHECK:     AddressAlign:    0x1
// CHECK:     Content:         '00000000'
// CHECK:   - Name:            .prgm.mem
// CHECK:     Type:            SHT_PROGBITS
// CHECK:     Flags:           [ SHF_ALLOC ]
// CHECK:     Address:         0x3060000
// CHECK:     AddressAlign:    0x1
// CHECK:     Content:         '00000000'
// CHECK:   - Name:            '.ssmast (1)'
// CHECK:     Type:            SHT_PROGBITS
// CHECK:     Flags:           [ SHF_ALLOC ]
// CHECK:     Address:         0x307F000
// CHECK:     AddressAlign:    0x1
// CHECK:     Content:         '00000000'
// CHECK:   - Name:            '.ssslve (1)'
// CHECK:     Type:            SHT_PROGBITS
// CHECK:     Flags:           [ SHF_ALLOC ]
// CHECK:     Address:         0x307F100
// CHECK:     AddressAlign:    0x1
// CHECK:     Content:         '00000000'
// CHECK:   - Name:            '.sspckt (1)'
// CHECK:     Type:            SHT_PROGBITS
// CHECK:     Flags:           [ SHF_ALLOC ]
// CHECK:     Address:         0x307F200
// CHECK:     AddressAlign:    0x1
// CHECK:     Content:         '00000000'
// CHECK:   - Name:            '.data.mem (1)'
// CHECK:     Type:            SHT_PROGBITS
// CHECK:     Flags:           [ SHF_ALLOC ]
// CHECK:     Address:         0x3080000
// CHECK:     AddressAlign:    0x1
// CHECK:     Content:         '00000000'
// CHECK:   - Type:            SHT_PROGBITS
// CHECK:     Flags:           [ SHF_ALLOC ]
// CHECK:     Address:         0x3080480
// CHECK:     AddressAlign:    0x1
// CHECK:     Content:         '70030000'
// CHECK:   - Name:            '.sdma.bd (2)'
// CHECK:     Type:            SHT_PROGBITS
// CHECK:     Flags:           [ SHF_ALLOC ]
// CHECK:     Address:         0x309D000
// CHECK:     AddressAlign:    0x1
// CHECK:     Content:         00013D00
// CHECK:   - Type:            SHT_PROGBITS
// CHECK:     Flags:           [ SHF_ALLOC ]
// CHECK:     Address:         0x309D020
// CHECK:     AddressAlign:    0x1
// CHECK:     Content:         10017D00
// CHECK:   - Type:            SHT_PROGBITS
// CHECK:     Flags:           [ SHF_ALLOC ]
// CHECK:     Address:         0x309D040
// CHECK:     AddressAlign:    0x1
// CHECK:     Content:         0801AF00
// CHECK:   - Type:            SHT_PROGBITS
// CHECK:     Flags:           [ SHF_ALLOC ]
// CHECK:     Address:         0x309D060
// CHECK:     AddressAlign:    0x1
// CHECK:     Content:         1801EF00
// CHECK:   - Name:            '.tdma.ctl (1)'
// CHECK:     Type:            SHT_PROGBITS
// CHECK:     Flags:           [ SHF_ALLOC ]
// CHECK:     Address:         0x309DE00
// CHECK:     AddressAlign:    0x1
// CHECK:     Content:         '01000000'
// CHECK:   - Name:            '.prgm.mem (1)'
// CHECK:     Type:            SHT_PROGBITS
// CHECK:     Flags:           [ SHF_ALLOC ]
// CHECK:     Address:         0x30A0000
// CHECK:     AddressAlign:    0x1
// CHECK:     Content:         0B00644A
// CHECK:   - Type:            SHT_PROGBITS
// CHECK:     Flags:           [ SHF_ALLOC ]
// CHECK:     Address:         0x30A0120
// CHECK:     AddressAlign:    0x1
// CHECK:     Content:         '03008240'
// CHECK:   - Type:            SHT_PROGBITS
// CHECK:     Flags:           [ SHF_ALLOC ]
// CHECK:     Address:         0x30A0340
// CHECK:     AddressAlign:    0x1
// CHECK:     Content:         '03307614'
// CHECK:   - Type:            SHT_PROGBITS
// CHECK:     Flags:           [ SHF_ALLOC ]
// CHECK:     Address:         0x30A0370
// CHECK:     AddressAlign:    0x1
// CHECK:     Content:         '77401008'
// CHECK:   - Type:            SHT_PROGBITS
// CHECK:     Flags:           [ SHF_ALLOC ]
// CHECK:     Address:         0x30A0450
// CHECK:     AddressAlign:    0x1
// CHECK:     Content:         3FA004AA
// CHECK:   - Name:            '.ssmast (2)'
// CHECK:     Type:            SHT_PROGBITS
// CHECK:     Flags:           [ SHF_ALLOC ]
// CHECK:     Address:         0x30BF000
// CHECK:     AddressAlign:    0x1
// CHECK:     Content:         '00000000'
// CHECK:   - Name:            '.ssslve (2)'
// CHECK:     Type:            SHT_PROGBITS
// CHECK:     Flags:           [ SHF_ALLOC ]
// CHECK:     Address:         0x30BF100
// CHECK:     AddressAlign:    0x1
// CHECK:     Content:         '00000000'
// CHECK:   - Name:            '.sspckt (2)'
// CHECK:     Type:            SHT_PROGBITS
// CHECK:     Flags:           [ SHF_ALLOC ]
// CHECK:     Address:         0x30BF200
// CHECK:     AddressAlign:    0x1
// CHECK:     Content:         '00000000'

module {
  AIE.device(xcvc1902) {
    %tile_6_0 = AIE.tile(6, 0)
    %tile_6_1 = AIE.tile(6, 1)
    %tile_6_2 = AIE.tile(6, 2)
    AIE.flow(%tile_6_0, DMA : 0, %tile_6_2, DMA : 0)
    AIE.flow(%tile_6_0, DMA : 1, %tile_6_2, DMA : 1)
    AIE.flow(%tile_6_2, DMA : 0, %tile_6_0, DMA : 0)
    AIE.flow(%tile_6_2, DMA : 1, %tile_6_0, DMA : 1)
    %buffer_6_2 = AIE.buffer(%tile_6_2) {address = 1024 : i32, sym_name = "ping_in"} : memref<8xi32>
    %buffer_6_2_0 = AIE.buffer(%tile_6_2) {address = 1056 : i32, sym_name = "ping_out"} : memref<8xi32>
    %buffer_6_2_1 = AIE.buffer(%tile_6_2) {address = 1088 : i32, sym_name = "pong_in"} : memref<8xi32>
    %buffer_6_2_2 = AIE.buffer(%tile_6_2) {address = 1120 : i32, sym_name = "pong_out"} : memref<8xi32>
    %lock_6_2 = AIE.lock(%tile_6_2, 0)
    %lock_6_2_3 = AIE.lock(%tile_6_2, 1)
    %lock_6_2_4 = AIE.lock(%tile_6_2, 2)
    %lock_6_2_5 = AIE.lock(%tile_6_2, 3)
    %mem_6_2 = AIE.mem(%tile_6_2) {
      %0 = AIE.dmaStart(S2MM, 0, ^bb2, ^bb1)
    ^bb1:  // pred: ^bb0
      %1 = AIE.dmaStart(MM2S, 0, ^bb4, ^bb6)
    ^bb2:  // 2 preds: ^bb0, ^bb3
      AIE.useLock(%lock_6_2, Acquire, 0)
      AIE.dmaBd(<%buffer_6_2 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%lock_6_2, Release, 1)
      AIE.nextBd ^bb3
    ^bb3:  // pred: ^bb2
      AIE.useLock(%lock_6_2_3, Acquire, 0)
      AIE.dmaBd(<%buffer_6_2_1 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%lock_6_2_3, Release, 1)
      AIE.nextBd ^bb2
    ^bb4:  // 2 preds: ^bb1, ^bb5
      AIE.useLock(%lock_6_2_4, Acquire, 1)
      AIE.dmaBd(<%buffer_6_2_0 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%lock_6_2_4, Release, 0)
      AIE.nextBd ^bb5
    ^bb5:  // pred: ^bb4
      AIE.useLock(%lock_6_2_5, Acquire, 1)
      AIE.dmaBd(<%buffer_6_2_2 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%lock_6_2_5, Release, 0)
      AIE.nextBd ^bb4
    ^bb6:  // pred: ^bb1
      AIE.end
    }
    %core_6_2 = AIE.core(%tile_6_2) {
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_i32 = arith.constant 1 : i32
      AIE.useLock(%lock_6_2, Acquire, 1)
      AIE.useLock(%lock_6_2_4, Acquire, 0)
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c8 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %2 = memref.load %buffer_6_2[%0] : memref<8xi32>
      %3 = arith.addi %2, %c1_i32 : i32
      memref.store %3, %buffer_6_2_0[%0] : memref<8xi32>
      %4 = arith.addi %0, %c1 : index
      cf.br ^bb1(%4 : index)
    ^bb3:  // pred: ^bb1
      AIE.useLock(%lock_6_2, Release, 0)
      AIE.useLock(%lock_6_2_4, Release, 1)
      AIE.useLock(%lock_6_2_3, Acquire, 1)
      AIE.useLock(%lock_6_2_5, Acquire, 0)
      cf.br ^bb4(%c0 : index)
    ^bb4(%5: index):  // 2 preds: ^bb3, ^bb5
      %6 = arith.cmpi slt, %5, %c8 : index
      cf.cond_br %6, ^bb5, ^bb6
    ^bb5:  // pred: ^bb4
      %7 = memref.load %buffer_6_2_1[%5] : memref<8xi32>
      %8 = arith.addi %7, %c1_i32 : i32
      memref.store %8, %buffer_6_2_2[%5] : memref<8xi32>
      %9 = arith.addi %5, %c1 : index
      cf.br ^bb4(%9 : index)
    ^bb6:  // pred: ^bb4
      AIE.useLock(%lock_6_2_3, Release, 0)
      AIE.useLock(%lock_6_2_5, Release, 1)
      AIE.end
    }
  }
}
