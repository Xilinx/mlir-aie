// RUN: aiecc.py --sysroot=%VITIS_SYSROOT% %s -I%aie_runtime_lib% %aie_runtime_lib%/test_library.cpp %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf

module @test_broadcast_packet {
  
  %t72 = AIE.tile(7, 2)
  
  %t63 = AIE.tile(6, 3)
  %t64 = AIE.tile(6, 4)
  %t73 = AIE.tile(7, 3)
  %t74 = AIE.tile(7, 4)

  %buf72_0 = AIE.buffer(%t72) {sym_name = "buf72_0"} : memref<1024xi32>
  %buf72_1 = AIE.buffer(%t72) {sym_name = "buf72_1"} : memref<1024xi32>

  %buf63_0 = AIE.buffer(%t63) {sym_name = "buf63_0"} : memref<1024xi32>
  %buf64_0 = AIE.buffer(%t64) {sym_name = "buf64_0"} : memref<1024xi32>

  %buf73_0 = AIE.buffer(%t73) {sym_name = "buf73_0"} : memref<1024xi32>
  %buf74_0 = AIE.buffer(%t74) {sym_name = "buf74_0"} : memref<1024xi32>

  AIE.broadcast_packet(%t72, "DMA" : 0){
    AIE.bp_id(0x0){
      AIE.bp_dest<%t73, "DMA" : 0>
      AIE.bp_dest<%t63, "DMA" : 0>
    }
    AIE.bp_id(0x1){
      AIE.bp_dest<%t74, "DMA" : 0>
      AIE.bp_dest<%t64, "DMA" : 0>
    }
  }

  %m72 = AIE.mem(%t72) {
    %lock72_4 = AIE.lock(%t72, 4)
    %lock72_5 = AIE.lock(%t72, 5)
    AIE.dmaStart("MM2S0", ^bd4, ^end)
    ^bd4:
      AIE.useLock(%lock72_4, "Acquire", 1)
      AIE.dmaBdPacket(0x0, 0x0)
      AIE.dmaBd(<%buf72_0 : memref<1024xi32>, 0, 1024>, 0)
      AIE.useLock(%lock72_4, "Release", 0)
      cf.br ^bd5
    ^bd5:
      AIE.useLock(%lock72_5, "Acquire", 1)
      AIE.dmaBdPacket(0x1, 0x1)
      AIE.dmaBd(<%buf72_1 : memref<1024xi32>, 0, 1024>, 0)
      AIE.useLock(%lock72_5, "Release", 0)
      cf.br ^bd4
    ^end:
      AIE.end
  }

  %lock63_0 = AIE.lock(%t63, 0)
  %m63 = AIE.mem(%t63)  {
  AIE.dmaStart("S2MM0", ^bd0, ^end)
  ^bd0: 
    AIE.useLock(%lock63_0, Acquire, 0)
    AIE.dmaBd(<%buf63_0 : memref<1024xi32>, 0, 1024>, 0)
    AIE.useLock(%lock63_0, Release, 1)
    cf.br ^bd0
  ^end: 
    AIE.end
  }


  %lock64_0 = AIE.lock(%t64, 0)
  %m64 = AIE.mem(%t64)  {
  AIE.dmaStart("S2MM0", ^bd0, ^end)
  ^bd0: 
    AIE.useLock(%lock64_0, Acquire, 0)
    AIE.dmaBd(<%buf64_0 : memref<1024xi32>, 0, 1024>, 0)
    AIE.useLock(%lock64_0, Release, 1)
    cf.br ^bd0
  ^end: 
    AIE.end
  }

 
  %lock73_0 = AIE.lock(%t73, 0)
  %m73 = AIE.mem(%t73)  {
  AIE.dmaStart("S2MM0", ^bd0, ^end)
  ^bd0: 
    AIE.useLock(%lock73_0, Acquire, 0)
    AIE.dmaBd(<%buf73_0 : memref<1024xi32>, 0, 1024>, 0)
    AIE.useLock(%lock73_0, Release, 1)
    cf.br ^bd0
  ^end: 
    AIE.end
  }

  %lock74_0 = AIE.lock(%t74, 0)
  %m74 = AIE.mem(%t74)  {
  
  AIE.dmaStart("S2MM0", ^bd0, ^end)
  ^bd0: 
    AIE.useLock(%lock74_0, Acquire, 0)
    AIE.dmaBd(<%buf74_0 : memref<1024xi32>, 0, 1024>, 0)
    AIE.useLock(%lock74_0, Release, 1)
    cf.br ^bd0
  ^end: 
    AIE.end
  }

}