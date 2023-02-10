// (c) 2023 SAFARI Research Group at ETH Zurich, Gagandeep Singh, D-ITET   
  
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

module @hdiff_multi_AIE{

  %t72 = AIE.tile(7, 2)
  %t71 = AIE.tile(7, 1)
  %t70 = AIE.tile(7, 0)
  
  %lock71_14 = AIE.lock(%t71, 14) { sym_name = "lock71_14" }
  %lock72_14 = AIE.lock(%t72, 14) { sym_name = "lock72_14" }

  %obj_fifo_in = AIE.objectFifo.createObjectFifo(%t70, {%t71,%t72}, 6) {sym_name = "" }: !AIE.objectFifo<memref<256xf32>>
  %obj_fifo_out_lap = AIE.objectFifo.createObjectFifo(%t71, {%t72}, 4){sym_name = "obj_out_lap" } : !AIE.objectFifo<memref<256xf32>>
  %obj_fifo_out_flux = AIE.objectFifo.createObjectFifo(%t72, {%t70}, 2){sym_name = "obj_out_flux" } : !AIE.objectFifo<memref<256xf32>>
   // DDR buffer
  %ext_buffer_in0  = AIE.external_buffer  {sym_name = "ddr_test_buffer_in0"}: memref<1536 x f32>
  %ext_buffer_out = AIE.external_buffer  {sym_name = "ddr_test_buffer_out"}: memref<512 x f32>
      
  // Register the external memory pointers to the object FIFOs.
  AIE.objectFifo.registerExternalBuffers(%t70, %obj_fifo_in : !AIE.objectFifo<memref<256xf32>>, {%ext_buffer_in0}) : (memref<1536xf32>)
  AIE.objectFifo.registerExternalBuffers(%t70, %obj_fifo_out_flux : !AIE.objectFifo<memref<256xf32>>, {%ext_buffer_out}) : (memref<512xf32>)

  func.func private @hdiff_lap_fp32(%AL: memref<256xf32>,%BL: memref<256xf32>, %CL:  memref<256xf32>, %DL: memref<256xf32>, %EL:  memref<256xf32>,  %OLL1: memref<256xf32>,  %OLL2: memref<256xf32>,  %OLL3: memref<256xf32>,  %OLL4: memref<256xf32>) -> ()
  
  %c13 = AIE.core(%t71) { 
    
    %lb = arith.constant 0 : index
    %ub = arith.constant 2: index
    %step = arith.constant 1 : index
    AIE.useLock(%lock71_14, "Acquire", 0) // start the timer
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%obj_fifo_in : !AIE.objectFifo<memref<256xf32>>, 5) : !AIE.objectFifoSubview<memref<256xf32>>
      %row0 = AIE.objectFifo.subview.access %obj_in_subview[0] : !AIE.objectFifoSubview<memref<256xf32>> -> memref<256xf32>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[1] : !AIE.objectFifoSubview<memref<256xf32>> -> memref<256xf32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[2] : !AIE.objectFifoSubview<memref<256xf32>> -> memref<256xf32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xf32>> -> memref<256xf32>
      %row4 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xf32>> -> memref<256xf32>


      %obj_out_subview_lap = AIE.objectFifo.acquire<Produce>(%obj_fifo_out_lap : !AIE.objectFifo<memref<256xf32>>, 4 ) : !AIE.objectFifoSubview<memref<256xf32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xf32>> -> memref<256xf32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xf32>> -> memref<256xf32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xf32>> -> memref<256xf32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xf32>> -> memref<256xf32>

      func.call @hdiff_lap_fp32(%row0,%row1,%row2,%row3,%row4,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4) : (memref<256xf32>,memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>,  memref<256xf32>,  memref<256xf32>,  memref<256xf32>,  memref<256xf32>) -> ()
      AIE.objectFifo.release<Produce>(%obj_fifo_out_lap : !AIE.objectFifo<memref<256xf32>>, 1)
      AIE.objectFifo.release<Produce>(%obj_fifo_out_lap : !AIE.objectFifo<memref<256xf32>>, 1)
      AIE.objectFifo.release<Produce>(%obj_fifo_out_lap : !AIE.objectFifo<memref<256xf32>>, 1)
      AIE.objectFifo.release<Produce>(%obj_fifo_out_lap : !AIE.objectFifo<memref<256xf32>>, 1)
      AIE.objectFifo.release<Consume>(%obj_fifo_in : !AIE.objectFifo<memref<256xf32>>, 1)

    }
    AIE.objectFifo.release<Consume>(%obj_fifo_in : !AIE.objectFifo<memref<256xf32>>, 4)

    AIE.end
  } { link_with="hdiff_lap_fp32.o" }

func.func private @hdiff_flux_fp32(%AF: memref<256xf32>,%BF: memref<256xf32>, %CF:  memref<256xf32>,   %OLF1: memref<256xf32>,  %OLF2: memref<256xf32>,  %OLF3: memref<256xf32>,  %OLF4: memref<256xf32>,  %OF: memref<256xf32>) -> ()
  
  %c14 = AIE.core(%t72) { 
    
    %lb = arith.constant 0 : index
    %ub = arith.constant 2: index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%obj_fifo_in : !AIE.objectFifo<memref<256xf32>>, 5) : !AIE.objectFifoSubview<memref<256xf32>>
      
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[1] : !AIE.objectFifoSubview<memref<256xf32>> -> memref<256xf32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[2] : !AIE.objectFifoSubview<memref<256xf32>> -> memref<256xf32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xf32>> -> memref<256xf32>
      
      %obj_out_subview_lap = AIE.objectFifo.acquire<Consume>(%obj_fifo_out_lap : !AIE.objectFifo<memref<256xf32>>, 4) : !AIE.objectFifoSubview<memref<256xf32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xf32>> -> memref<256xf32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xf32>> -> memref<256xf32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xf32>> -> memref<256xf32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xf32>> -> memref<256xf32>

      %obj_out_subview_flux = AIE.objectFifo.acquire<Produce>(%obj_fifo_out_flux : !AIE.objectFifo<memref<256xf32>>, 1) : !AIE.objectFifoSubview<memref<256xf32>>
      %obj_out_flux = AIE.objectFifo.subview.access %obj_out_subview_flux[0] : !AIE.objectFifoSubview<memref<256xf32>> -> memref<256xf32>


      func.call @hdiff_flux_fp32(%row1,%row2,%row3,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4, %obj_out_flux ) : (memref<256xf32>,memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>,  memref<256xf32>,  memref<256xf32>) -> ()
      AIE.objectFifo.release<Consume>(%obj_fifo_out_lap : !AIE.objectFifo<memref<256xf32>>, 4)
      AIE.objectFifo.release<Produce>(%obj_fifo_out_flux : !AIE.objectFifo<memref<256xf32>>, 1)
      AIE.objectFifo.release<Consume>(%obj_fifo_in : !AIE.objectFifo<memref<256xf32>>, 1)
    }
    AIE.useLock(%lock72_14, "Acquire", 0) // stop the timer
    AIE.objectFifo.release<Consume>(%obj_fifo_in : !AIE.objectFifo<memref<256xf32>>, 4)

    AIE.end
  } { link_with="hdiff_flux_fp32.o" }



}

