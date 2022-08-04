// REQUIRES: valid_xchess_license
// RUN: xchesscc -p me -P ${CARDANO}/data/cervino/lib -c %S/kernel.cc
// RUN: aiecc.py --sysroot=%VITIS_SYSROOT% %s -I%aie_runtime_lib% %aie_runtime_lib%/test_library.cpp %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf

module @MM_2x2 {
  
  %t60 = AIE.tile(6, 0)
  %t63 = AIE.tile(6, 3)
  %t64 = AIE.tile(6, 4)

  %t70 = AIE.tile(7, 0)
  %t73 = AIE.tile(7, 3)
  %t74 = AIE.tile(7, 4)

  %buffer0 = AIE.external_buffer 0x020100000000 : memref<1024 x i32>     //LHS_tile0
  %buffer1 = AIE.external_buffer 0x020100001000 : memref<1024 x i32>     //LHS_tile1
  %buffer2 = AIE.external_buffer 0x020100002000 : memref<1024 x i32>     //RHS_tile0
  %buffer3 = AIE.external_buffer 0x020100003000 : memref<1024 x i32>     //RHS_tile1
  %buffer4 = AIE.external_buffer 0x020100004000 : memref<1024 x i32>     //RHS_tile2
  %buffer5 = AIE.external_buffer 0x020100005000 : memref<1024 x i32>     //RHS_tile3
  %buffer6 = AIE.external_buffer 0x020100006000 : memref<1025 x i32>     //Out_tile0
  %buffer7 = AIE.external_buffer 0x020100008000 : memref<1025 x i32>     //Out_tile1


  %buf63_0 = AIE.buffer(%t63) {sym_name = "buf63_0"} : memref<1024xi32>  //LHS_tile0
  %buf63_1 = AIE.buffer(%t63) {sym_name = "buf63_1"} : memref<1024xi32>  //RHS_tile0
  %buf63_2 = AIE.buffer(%t63) {sym_name = "buf63_2"} : memref<1024xi32>  //Accumulator
  %buf63_3 = AIE.buffer(%t63) {sym_name = "buf63_3"} : memref<1024xi32>  //Sub_sum0
  %buf64_0 = AIE.buffer(%t64) {sym_name = "buf64_0"} : memref<1024xi32>  //LHS_tile1
  %buf64_1 = AIE.buffer(%t64) {sym_name = "buf64_1"} : memref<1024xi32>  //RHS_tile1
  %buf64_2 = AIE.buffer(%t64) {sym_name = "buf64_2"} : memref<1024xi32>  //Out_tile0

  %buf73_0 = AIE.buffer(%t73) {sym_name = "buf73_0"} : memref<1024xi32>  //LHS_tile0
  %buf73_1 = AIE.buffer(%t73) {sym_name = "buf73_1"} : memref<1024xi32>  //RHS_tile2
  %buf73_2 = AIE.buffer(%t73) {sym_name = "buf73_2"} : memref<1024xi32>  //Accumulator
  %buf73_3 = AIE.buffer(%t73) {sym_name = "buf73_3"} : memref<1024xi32>  //Sub_sum1
  %buf74_0 = AIE.buffer(%t74) {sym_name = "buf74_0"} : memref<1024xi32>  //LHS_tile1
  %buf74_1 = AIE.buffer(%t74) {sym_name = "buf74_1"} : memref<1024xi32>  //RHS_tile3
  %buf74_2 = AIE.buffer(%t74) {sym_name = "buf74_2"} : memref<1024xi32>  //Out_tile1

  AIE.broadcast_packet(%t60, DMA : 0) {
    AIE.bp_id(0) {
      AIE.bp_dest<%t63, DMA : 0>
      AIE.bp_dest<%t73, DMA : 0>
    }
    AIE.bp_id(1) {
      AIE.bp_dest<%t64, DMA : 0>
      AIE.bp_dest<%t74, DMA : 0>
    }
  }

  AIE.broadcast_packet(%t60, DMA : 1) {
    AIE.bp_id(2) {
      AIE.bp_dest<%t63, DMA : 1>
    }
    AIE.bp_id(3) {
      AIE.bp_dest<%t64, DMA : 1>
    }
  }

  AIE.broadcast_packet(%t70, DMA : 0) {
    AIE.bp_id(4) {
      AIE.bp_dest<%t73, DMA : 1>
    }
    AIE.bp_id(5) {
      AIE.bp_dest<%t74, DMA : 1>
    }
  }

  AIE.broadcast_packet(%t64, DMA : 0) {
    AIE.bp_id(6) {
      AIE.bp_dest<%t70, DMA : 0>
    }
  }

  AIE.broadcast_packet(%t74, DMA : 0) {
    AIE.bp_id(7) {
      AIE.bp_dest<%t70, DMA : 0>
    }
  }

  %lock60_0 = AIE.lock(%t60, 0)
  %lock60_1 = AIE.lock(%t60, 1)
  %lock60_2 = AIE.lock(%t60, 2)
  %lock60_3 = AIE.lock(%t60, 3)
  

  %dma60 = AIE.shimDMA(%t60) {
    AIE.dmaStart("MM2S0", ^bd4, ^dma2)
    ^dma2:
        AIE.dmaStart("MM2S1", ^bd6, ^end)
    ^bd4:
      AIE.useLock(%lock60_0, "Acquire", 1)
      AIE.dmaBdPacket(0x0, 0x0)
      AIE.dmaBd(<%buffer0 : memref<1024xi32>, 0, 1024>, 0)    //send LHS_tile0 with Pack_ID=0
      AIE.useLock(%lock60_0, "Release", 0)
      cf.br ^bd5
    ^bd5:
      AIE.useLock(%lock60_1, "Acquire", 1)
      AIE.dmaBdPacket(0x1, 0x1)
      AIE.dmaBd(<%buffer1 : memref<1024xi32>, 0, 1024>, 0)    //send LHS_tile1 with Pack_ID=1
      AIE.useLock(%lock60_1, "Release", 0)
      cf.br ^bd4
    ^bd6:
      AIE.useLock(%lock60_2, "Acquire", 1)
      AIE.dmaBdPacket(0x2, 0x2)
      AIE.dmaBd(<%buffer2 : memref<1024xi32>, 0, 1024>, 0)    //send RHS_tile0 with Pack_ID=2
      AIE.useLock(%lock60_2, "Release", 0)
      cf.br ^bd7
    ^bd7:
      AIE.useLock(%lock60_3, "Acquire", 1)
      AIE.dmaBdPacket(0x3, 0x3)
      AIE.dmaBd(<%buffer3 : memref<1024xi32>, 0, 1024>, 0)    //send RHS_tile1 with Pack_ID=3
      AIE.useLock(%lock60_3, "Release", 0)
      cf.br ^bd6
    ^end:
      AIE.end
  }

  %lock70_0 = AIE.lock(%t70, 0)
  %lock70_1 = AIE.lock(%t70, 1)

  %dma70 = AIE.shimDMA(%t70) {
    AIE.dmaStart("MM2S0", ^bd4, ^dma2)
    ^dma2:
        AIE.dmaStart("S2MM0", ^bd6, ^end)
    ^bd4:
      AIE.useLock(%lock70_0, "Acquire", 1)
      AIE.dmaBdPacket(0x4, 0x4)
      AIE.dmaBd(<%buffer4 : memref<1024xi32>, 0, 1024>, 0)    //send RHS_tile2 with Pack_ID=4
      AIE.useLock(%lock70_0, "Release", 0)
      cf.br ^bd5
    ^bd5:
      AIE.useLock(%lock70_1, "Acquire", 1)
      AIE.dmaBdPacket(0x5, 0x5)
      AIE.dmaBd(<%buffer5 : memref<1024xi32>, 0, 1024>, 0)    //send RHS_tile3 with Pack_ID=5
      AIE.useLock(%lock70_1, "Release", 0)
      cf.br ^bd4
    ^bd6:
      AIE.dmaBd(<%buffer6 : memref<1025xi32>, 0, 1025>, 0)    //send Out_tile0 with Pack_ID=6
      cf.br ^bd7
    ^bd7:
      AIE.dmaBd(<%buffer7 : memref<1025xi32>, 0, 1025>, 0)    //send Out_tile1 with Pack_ID=7
      cf.br ^bd6
    ^end:
      AIE.end
  }


  %lock63_0 = AIE.lock(%t63, 0)
  %lock63_1 = AIE.lock(%t63, 1)
  %m63 = AIE.mem(%t63)  {
  AIE.dmaStart("S2MM0", ^bd0, ^dma0)
  ^dma0:
    AIE.dmaStart("S2MM1", ^bd1, ^end)
  ^bd0: 
    AIE.useLock(%lock63_0, Acquire, 0)
    AIE.dmaBd(<%buf63_0 : memref<1024xi32>, 0, 1024>, 0)
    AIE.useLock(%lock63_0, Release, 1)
    cf.br ^bd0
  ^bd1: 
    AIE.useLock(%lock63_1, Acquire, 0)
    AIE.dmaBd(<%buf63_1 : memref<1024xi32>, 0, 1024>, 0)
    AIE.useLock(%lock63_1, Release, 1)
    cf.br ^bd1
  ^end: 
    AIE.end
  }


  %lock64_0 = AIE.lock(%t64, 0)
  %lock64_1 = AIE.lock(%t64, 1)
  %m64 = AIE.mem(%t64)  {
  AIE.dmaStart("S2MM0", ^bd0, ^dma0)
  ^dma0:
    AIE.dmaStart("S2MM1", ^bd1, ^dma1)
  ^bd0: 
    AIE.useLock(%lock64_0, Acquire, 0)
    AIE.dmaBd(<%buf64_0 : memref<1024xi32>, 0, 1024>, 0)
    AIE.useLock(%lock64_0, Release, 1)
    cf.br ^bd0
  ^bd1: 
    AIE.useLock(%lock64_1, Acquire, 0)
    AIE.dmaBd(<%buf64_1 : memref<1024xi32>, 0, 1024>, 0)
    AIE.useLock(%lock64_1, Release, 1)
    cf.br ^bd1
  ^dma1:
    AIE.dmaStart("MM2S0", ^bd2, ^end)
  ^bd2:
    AIE.useLock(%lock64_2, Acquire, 1)
    AIE.dmaBdPacket(0, 6)
    AIE.dmaBd(<%buf64_2 : memref<1024xi32>, 0, 1024>, 0)
    AIE.useLock(%lock64_2, Release, 0)
    cf.br ^bd2
  ^end: 
    AIE.end
  }

  func.func private @extern_kernel(%A: memref<1024xi32>, %B: memref<1024xi32>, %acc: memref<1024xi32>, %C: memref<1024xi32>) -> ()


  %lock63_3 = AIE.lock(%t63, 3)
  %core63 = AIE.core(%t63) { 
    AIE.useLock(%lock63_0, "Acquire", 1)
    AIE.useLock(%lock63_1, "Acquire", 1)
    AIE.useLock(%lock63_3, "Acquire", 0)
    func.call @extern_kernel(%buf63_0, %buf63_1, %buf63_2, %buf63_3) : (memref<1024xi32>, memref<1024xi32>, memref<1024xi32>, memref<1024xi32>) -> ()
    AIE.useLock(%lock63_3, "Release", 1)
    AIE.useLock(%lock63_1, "Release", 0)
    AIE.useLock(%lock63_0, "Release", 0)
    
    AIE.end
  } { link_with="kernel.o" }


  %lock64_2 = AIE.lock(%t64, 2)
  %core64 = AIE.core(%t64) { 
    AIE.useLock(%lock63_3, "Acquire", 1)
    AIE.useLock(%lock64_0, "Acquire", 1)
    AIE.useLock(%lock64_1, "Acquire", 1)
    AIE.useLock(%lock64_2, "Acquire", 0)
    func.call @extern_kernel(%buf64_0, %buf64_1, %buf63_3, %buf64_2) : (memref<1024xi32>, memref<1024xi32>, memref<1024xi32>, memref<1024xi32>) -> ()
    AIE.useLock(%lock64_2, "Release", 1)
    AIE.useLock(%lock64_1, "Release", 0)
    AIE.useLock(%lock64_0, "Release", 0)
    AIE.useLock(%lock63_3, "Release", 0)
    AIE.end
  } { link_with="kernel.o" }


  %lock73_0 = AIE.lock(%t73, 0)
  %lock73_1 = AIE.lock(%t73, 1)
  
  %m73 = AIE.mem(%t73)  {
  AIE.dmaStart("S2MM0", ^bd0, ^dma0)
  ^dma0:
    AIE.dmaStart("S2MM1", ^bd1, ^end)
  ^bd0: 
    AIE.useLock(%lock73_0, Acquire, 0)
    AIE.dmaBd(<%buf73_0 : memref<1024xi32>, 0, 1024>, 0)
    AIE.useLock(%lock73_0, Release, 1)
    cf.br ^bd0
  ^bd1: 
    AIE.useLock(%lock73_1, Acquire, 0)
    AIE.dmaBd(<%buf73_1 : memref<1024xi32>, 0, 1024>, 0)
    AIE.useLock(%lock73_1, Release, 1)
    cf.br ^bd1
  ^end: 
    AIE.end
  }


  %lock74_0 = AIE.lock(%t74, 0)
  %lock74_1 = AIE.lock(%t74, 1)
  %m74 = AIE.mem(%t74)  {
  
  AIE.dmaStart("S2MM0", ^bd0, ^dma0)
  ^dma0:
    AIE.dmaStart("S2MM1", ^bd1, ^dma1)
  ^bd0: 
    AIE.useLock(%lock74_0, Acquire, 0)
    AIE.dmaBd(<%buf74_0 : memref<1024xi32>, 0, 1024>, 0)
    AIE.useLock(%lock74_0, Release, 1)
    cf.br ^bd0
  ^bd1: 
    AIE.useLock(%lock74_1, Acquire, 0)
    AIE.dmaBd(<%buf74_1 : memref<1024xi32>, 0, 1024>, 0)
    AIE.useLock(%lock74_1, Release, 1)
    cf.br ^bd1
  ^dma1:
    AIE.dmaStart("MM2S0", ^bd2, ^end)
  ^bd2:
    AIE.useLock(%lock74_2, Acquire, 1)
    AIE.dmaBdPacket(0, 7)
    AIE.dmaBd(<%buf74_2 : memref<1024xi32>, 0, 1024>, 0)
    AIE.useLock(%lock74_2, Release, 0)
    cf.br ^bd2
  ^end: 
    AIE.end
  }


  %lock73_2 = AIE.lock(%t73, 2)
  %core73 = AIE.core(%t73) { 
    AIE.useLock(%lock73_0, "Acquire", 1)
    AIE.useLock(%lock73_1, "Acquire", 1)
    AIE.useLock(%lock73_2, "Acquire", 0)
    func.call @extern_kernel(%buf73_0, %buf73_1, %buf73_2, %buf73_3) : (memref<1024xi32>, memref<1024xi32>, memref<1024xi32>, memref<1024xi32>) -> ()
    AIE.useLock(%lock73_2, "Release", 1)
    AIE.useLock(%lock73_1, "Release", 0)
    AIE.useLock(%lock73_0, "Release", 0)   
    AIE.end
  } { link_with="kernel.o" }

  %lock74_2 = AIE.lock(%t74, 2)
  %core74 = AIE.core(%t74) { 
    AIE.useLock(%lock73_2, "Acquire", 1)
    AIE.useLock(%lock74_0, "Acquire", 1)
    AIE.useLock(%lock74_1, "Acquire", 1)
    AIE.useLock(%lock74_2, "Acquire", 0)
    func.call @extern_kernel(%buf74_0, %buf74_1, %buf73_3, %buf74_2) : (memref<1024xi32>, memref<1024xi32>, memref<1024xi32>, memref<1024xi32>) -> ()
    AIE.useLock(%lock74_2, "Release", 1)
    AIE.useLock(%lock74_1, "Release", 0)
    AIE.useLock(%lock74_0, "Release", 0)
    AIE.useLock(%lock73_2, "Release", 0)
    AIE.end
  } { link_with="kernel.o" }


}