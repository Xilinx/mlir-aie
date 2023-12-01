//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --pass-pipeline='builtin.module(aie-canonicalize-device,AIE.device(aie-assign-buffer-addresses),convert-scf-to-cf)' %s | aie-translate --aie-generate-airbin --oo=%T/airbin.elf | %LLVM_TOOLS_DIR/obj2yaml %T/airbin.elf | FileCheck %s

// CHECK-LABEL: --- !ELF
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
// CHECK:     Content:         '40010000'
// CHECK:   - Name:            .ssmast
// CHECK:     Type:            SHT_PROGBITS
// CHECK:     Flags:           [ SHF_ALLOC ]
// CHECK:     Address:         0x303F000
// CHECK:     AddressAlign:    0x1
// CHECK:     Content:         5C000000
// CHECK:   - Name:            .ssslve
// CHECK:     Type:            SHT_PROGBITS
// CHECK:     Flags:           [ SHF_ALLOC ]
// CHECK:     Address:         0x303F100
// CHECK:     AddressAlign:    0x1
// CHECK:     Content:         '60000000'
// CHECK:   - Name:            .sspckt
// CHECK:     Type:            SHT_PROGBITS
// CHECK:     Flags:           [ SHF_ALLOC ]
// CHECK:     Address:         0x303F200
// CHECK:     AddressAlign:    0x1
// CHECK:     Content:         '80010000'
// CHECK:   - Name:            .data.mem
// CHECK:     Type:            SHT_PROGBITS
// CHECK:     Flags:           [ SHF_ALLOC ]
// CHECK:     Address:         0x3040000
// CHECK:     AddressAlign:    0x1
// CHECK:     Content:         '00800000'
// CHECK:   - Name:            '.sdma.bd (1)'
// CHECK:     Type:            SHT_PROGBITS
// CHECK:     Flags:           [ SHF_ALLOC ]
// CHECK:     Address:         0x305D000
// CHECK:     AddressAlign:    0x1
// CHECK:     Content:         '00020000'
// CHECK:   - Name:            .tdma.ctl
// CHECK:     Type:            SHT_PROGBITS
// CHECK:     Flags:           [ SHF_ALLOC ]
// CHECK:     Address:         0x305DE00
// CHECK:     AddressAlign:    0x1
// CHECK:     Content:         '10000000'
// CHECK:   - Type:            SHT_PROGBITS
// CHECK:     Flags:           [ SHF_ALLOC ]
// CHECK:     Address:         0x305DE10
// CHECK:     AddressAlign:    0x1
// CHECK:     Content:         '10000000'
// CHECK:   - Name:            .prgm.mem
// CHECK:     Type:            SHT_PROGBITS
// CHECK:     Flags:           [ SHF_ALLOC ]
// CHECK:     Address:         0x3060000
// CHECK:     AddressAlign:    0x1
// CHECK:     Content:         '00400000'
// CHECK:   - Name:            '.ssmast (1)'
// CHECK:     Type:            SHT_PROGBITS
// CHECK:     Flags:           [ SHF_ALLOC ]
// CHECK:     Address:         0x307F000
// CHECK:     AddressAlign:    0x1
// CHECK:     Content:         '64000000'
// CHECK:   - Name:            '.ssslve (1)'
// CHECK:     Type:            SHT_PROGBITS
// CHECK:     Flags:           [ SHF_ALLOC ]
// CHECK:     Address:         0x307F100
// CHECK:     AddressAlign:    0x1
// CHECK:     Content:         6C000000
// CHECK:   - Name:            '.sspckt (1)'
// CHECK:     Type:            SHT_PROGBITS
// CHECK:     Flags:           [ SHF_ALLOC ]
// CHECK:     Address:         0x307F200
// CHECK:     AddressAlign:    0x1
// CHECK:     Content:         C0060000
// CHECK:   - Name:            '.data.mem (1)'
// CHECK:     Type:            SHT_PROGBITS
// CHECK:     Flags:           [ SHF_ALLOC ]
// CHECK:     Address:         0x3080000
// CHECK:     AddressAlign:    0x1
// CHECK:     Content:         '00800000'
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
// CHECK:     Content:         '0100000000000000000000000000000001000000020000000000000000000000'
// CHECK:   - Name:            '.prgm.mem (1)'
// CHECK:     Type:            SHT_PROGBITS
// CHECK:     Flags:           [ SHF_ALLOC ]
// CHECK:     Address:         0x30A0000
// CHECK:     AddressAlign:    0x1
// CHECK:     Content:         '00400000'
// CHECK:   - Name:            '.ssmast (2)'
// CHECK:     Type:            SHT_PROGBITS
// CHECK:     Flags:           [ SHF_ALLOC ]
// CHECK:     Address:         0x30BF000
// CHECK:     AddressAlign:    0x1
// CHECK:     Content:         '64000000'
// CHECK:   - Name:            '.ssslve (2)'
// CHECK:     Type:            SHT_PROGBITS
// CHECK:     Flags:           [ SHF_ALLOC ]
// CHECK:     Address:         0x30BF100
// CHECK:     AddressAlign:    0x1
// CHECK:     Content:         6C000000
// CHECK:   - Name:            '.sspckt (2)'
// CHECK:     Type:            SHT_PROGBITS
// CHECK:     Flags:           [ SHF_ALLOC ]
// CHECK:     Address:         0x30BF200
// CHECK:     AddressAlign:    0x1
// CHECK:     Content:         C0060000
// CHECK: ...

module {
  %t70 = AIE.tile(6, 0)
  %t71 = AIE.tile(6, 1)
  %t72 = AIE.tile(6, 2)

  AIE.flow(%t70, "DMA" : 0, %t72, "DMA" : 0)
  AIE.flow(%t70, "DMA" : 1, %t72, "DMA" : 1)
  AIE.flow(%t72, "DMA" : 0, %t70, "DMA" : 0)
  AIE.flow(%t72, "DMA" : 1, %t70, "DMA" : 1)

  %buf72_0 = AIE.buffer(%t72) { sym_name = "ping_in" } : memref<8xi32>
  %buf72_1 = AIE.buffer(%t72) { sym_name = "ping_out" } : memref<8xi32>
  %buf72_2 = AIE.buffer(%t72) { sym_name = "pong_in" } : memref<8xi32>
  %buf72_3 = AIE.buffer(%t72) { sym_name = "pong_out" } : memref<8xi32>

  %l72_0 = AIE.lock(%t72, 0)
  %l72_1 = AIE.lock(%t72, 1)
  %l72_2 = AIE.lock(%t72, 2)
  %l72_3 = AIE.lock(%t72, 3)

  %m72 = AIE.mem(%t72) {
      %srcDma = AIE.dmaStart(S2MM, 0, ^bd0, ^dma0)
    ^dma0:
      %dstDma = AIE.dmaStart(MM2S, 0, ^bd2, ^end)
    ^bd0:
      AIE.useLock(%l72_0, "Acquire", 0)
      AIE.dmaBd(<%buf72_0 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%l72_0, "Release", 1)
      AIE.nextBd ^bd1
    ^bd1:
      AIE.useLock(%l72_1, "Acquire", 0)
      AIE.dmaBd(<%buf72_2 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%l72_1, "Release", 1)
      AIE.nextBd ^bd0
    ^bd2:
      AIE.useLock(%l72_2, "Acquire", 1)
      AIE.dmaBd(<%buf72_1 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%l72_2, "Release", 0)
      AIE.nextBd ^bd3
    ^bd3:
      AIE.useLock(%l72_3, "Acquire", 1)
      AIE.dmaBd(<%buf72_3 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%l72_3, "Release", 0)
      AIE.nextBd ^bd2
    ^end:
      AIE.end
  }

  AIE.core(%t72) {
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_32 = arith.constant 1 : i32

    AIE.useLock(%l72_0, "Acquire", 1)
    AIE.useLock(%l72_2, "Acquire", 0)
    scf.for %arg3 = %c0 to %c8 step %c1 {
        %0 = memref.load %buf72_0[%arg3] : memref<8xi32>
        %1 = arith.addi %0, %c1_32 : i32
        memref.store %1, %buf72_1[%arg3] : memref<8xi32>
    }
    AIE.useLock(%l72_0, "Release", 0)
    AIE.useLock(%l72_2, "Release", 1)

    AIE.useLock(%l72_1, "Acquire", 1)
    AIE.useLock(%l72_3, "Acquire", 0)
    scf.for %arg4 = %c0 to %c8 step %c1 {
        %2 = memref.load %buf72_2[%arg4] : memref<8xi32>
        %3 = arith.addi %2, %c1_32 : i32
        memref.store %3, %buf72_3[%arg4] : memref<8xi32>
    }
    AIE.useLock(%l72_1, "Release", 0)
    AIE.useLock(%l72_3, "Release", 1)
    AIE.end

  }

}
