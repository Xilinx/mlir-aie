// RUN: aiecc.py --sysroot=/group/xrlabs/platforms/pynq_on_versal_vck190_hacked/vck190-sysroot %s -I%S/../../../runtime_lib %S/../../../runtime_lib/test_library.cpp %S/test.cpp -o test.elf

module {
  %t70 = AIE.tile(7, 0)
  %t71 = AIE.tile(7, 1)
  %t72 = AIE.tile(7, 2)

//  %buffer = AIE.external_buffer 0x20000 : memref<512 x i32>
  %buffer = AIE.external_buffer 0x020100004000 : memref<512 x i32>

  // Fixup
  %sw = AIE.switchbox(%t70) {
    AIE.connect<"South" : 3, "North" : 3>
  }
  %mux = AIE.shimmux(%t70) {
    AIE.connect<"DMA" : 0, "South": 3>
  }

  %dma = AIE.shimDMA(%t70) {
    %lock1 = AIE.lock(%t70, 1)

      AIE.dmaStart(MM2S0, ^bd0, ^end)

    ^bd0:
      AIE.useLock(%lock1, Acquire, 1, 0)
      AIE.dmaBd(<%buffer : memref<512 x i32>, 0, 512>, 0)
      AIE.useLock(%lock1, Release, 0, 0)
      br ^bd0
    ^end:
      AIE.end
  }

  AIE.flow(%t71, "South" : 3, %t72, "DMA" : 0)

  %buf72_0 = AIE.buffer(%t72) {sym_name = "buf72_0" } : memref<256xi32>
  %buf72_1 = AIE.buffer(%t72) {sym_name = "buf72_1" } : memref<256xi32>

  %l72_0 = AIE.lock(%t72, 0)
  %l72_1 = AIE.lock(%t72, 1)

  %m72 = AIE.mem(%t72) {
      %srcDma = AIE.dmaStart("S2MM0", ^bd0, ^end)
    ^bd0:
      AIE.useLock(%l72_0, "Acquire", 0, 0)
      AIE.dmaBd(<%buf72_0 : memref<256xi32>, 0, 256>, 0)
      AIE.useLock(%l72_0, "Release", 1, 0)
      br ^bd1
    ^bd1:
      AIE.useLock(%l72_1, "Acquire", 0, 0)
      AIE.dmaBd(<%buf72_1 : memref<256xi32>, 0, 256>, 0)
      AIE.useLock(%l72_1, "Release", 1, 0)
      br ^bd0
    ^end:
      AIE.end
  }

  


}
