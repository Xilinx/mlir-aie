//===- aieml_xcve2802.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Denis Ryndine
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// CDO for an AIE-ML Versal device (xcve2802): one stream from the shim DMA of
// the shim NOC tile (2, 0) through the memory tiles to the S2MM channel of the
// core tile (2, 3).

// RUN: aie-translate --aie-generate-cdo %s --cdo-debug=true | FileCheck %s

// The debug printer drops the upper 32 bits of an address on hosts with a
// 32-bit long, so these lines match the offset within the array only.
// Core tile (2, 3) lock 0 initial value.
// CHECK: (Write64): Address:  0x{{[0-9A-F]+}}0431F000 Data:  0x00000001
// Memory tile stream switches (2, 1) and (2, 2): master North:5, slave South:5.
// CHECK: (Write64): Address:  0x{{[0-9A-F]+}}041B0040 Data:  0x8000000C
// CHECK: (Write64): Address:  0x{{[0-9A-F]+}}041B0130 Data:  0x80000000
// CHECK: (Write64): Address:  0x{{[0-9A-F]+}}042B0040 Data:  0x8000000C
// CHECK: (Write64): Address:  0x{{[0-9A-F]+}}042B0130 Data:  0x80000000
// Core tile (2, 3) BD 0, then S2MM 0 start queue and enable.
// CHECK: (BlockWrite-DMAWriteCmd): Start Address: 0x{{[0-9A-F]+}}0431D000  Size: 6
// CHECK: (Write64): Address:  0x{{[0-9A-F]+}}0431DE04 Data:  0x00000000
// CHECK: (MaskWrite64): Address: 0x{{[0-9A-F]+}}0431DE00  Mask: 0x00000000  Data: 0x00000001

// The full 64-bit addresses, read from the binary: absolute from the Versal
// array base 0x20000000000, column shift 25, row shift 20. Each line is the
// command (105 DMA write, 107 mask write, 108 write) and its address.
// RUN: rm -rf %t && mkdir -p %t
// RUN: aie-translate --aie-generate-cdo --work-dir-path=%t %s
// RUN: %PYTHON -c "exec('import struct,sys\nw=[x for (x,) in struct.iter_unpack(\'<I\',open(sys.argv[1],\'rb\').read())]\ni=w[0]+1\nwhile i<len(w):\n h=w[i];n=h>>16&255;i+=1;e=n==255;n=w[i] if e else n;i+=e\n if h&255 in (5,7,8): print(\'{:03X} {:08X}{:08X}\'.format(h&4095,w[i],w[i+1]))\n i+=n')" %t/main_aie_cdo_init.bin | FileCheck %s --check-prefix=ADDR

// ADDR: 108 000002000431F000
// ADDR-NEXT: 108 000002000403F044
// ADDR-NEXT: 108 000002000403F114
// ADDR-NEXT: 108 00000200041B0040
// ADDR-NEXT: 108 00000200041B0130
// ADDR-NEXT: 108 00000200042B0040
// ADDR-NEXT: 108 00000200042B0130
// ADDR-NEXT: 108 000002000433F004
// ADDR-NEXT: 108 000002000433F128
// ADDR-NEXT: 107 000002000401F000
// ADDR-NEXT: 105 000002000431D000
// ADDR-NEXT: 108 000002000431DE04
// ADDR-NEXT: 107 000002000431DE00
// ADDR-EMPTY:

// The PDI and xclbin steps of aiecc stay NPU-only.
// RUN: not aiecc -n --get-pdi --tmpdir=%t/prj %s 2>&1 | FileCheck %s --check-prefix=PDI
// RUN: not aiecc -n --get-xclbin --tmpdir=%t/prj %s 2>&1 | FileCheck %s --check-prefix=PDI
// PDI: aiecc: --get-pdi/-xclbin are NPU-only, but 'main' is not an NPU device

module {
  aie.device(xcve2802) {
    %shim = aie.tile(2, 0)
    %mem1 = aie.tile(2, 1)
    %mem2 = aie.tile(2, 2)
    %core = aie.tile(2, 3)
    %lock = aie.lock(%core, 0) {init = 1 : i32}
    %buf = aie.buffer(%core) {address = 1024 : i32} : memref<16xi32>
    %dma = aie.mem(%core) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aie.use_lock(%lock, AcquireGreaterEqual, %one)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32}
      aie.use_lock(%lock, Release, %one)
      aie.next_bd ^end
    ^end:
      aie.end
    }
    %mux20 = aie.shim_mux(%shim) {
      aie.connect<DMA : 0, North : 3>
    }
    %sw20 = aie.switchbox(%shim) {
      aie.connect<South : 3, North : 5>
    }
    %sw21 = aie.switchbox(%mem1) {
      aie.connect<South : 5, North : 5>
    }
    %sw22 = aie.switchbox(%mem2) {
      aie.connect<South : 5, North : 5>
    }
    %sw23 = aie.switchbox(%core) {
      aie.connect<South : 5, DMA : 0>
    }
  }
}
