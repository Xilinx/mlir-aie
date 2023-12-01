//===- aie_fp32.mlir -------------------------------------------*- MLIR -*-===//
//
// (c) 2023 SAFARI Research Group at ETH Zurich, Gagandeep Singh, D-ITET
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
// 
//
//===----------------------------------------------------------------------===//


module @hdiff_multi_AIE{

  %t72 = AIE.tile(7, 2)
  %t71 = AIE.tile(7, 1)
  %t70 = AIE.tile(7, 0)
  
  %lock71_14 = AIE.lock(%t71, 14) { sym_name = "lock71_14" }
  %lock72_14 = AIE.lock(%t72, 14) { sym_name = "lock72_14" }

  AIE.objectfifo @obj_in (%t70, {%t71,%t72}, 6 : i32) : !AIE.objectfifo<memref<256xf32>>
  AIE.objectfifo @obj_out_lap (%t71, {%t72}, 4 : i32) : !AIE.objectfifo<memref<256xf32>>
  AIE.objectfifo @obj_out_flux (%t72, {%t70}, 2 : i32) : !AIE.objectfifo<memref<256xf32>>
   // DDR buffer
  %ext_buffer_in0  = AIE.external_buffer  {sym_name = "ddr_test_buffer_in0"}: memref<1536 x f32>
  %ext_buffer_out = AIE.external_buffer  {sym_name = "ddr_test_buffer_out"}: memref<512 x f32>
      
  // Register the external memory pointers to the object FIFOs.
  AIE.objectfifo.register_external_buffers @obj_in (%t70, {%ext_buffer_in0}) : (memref<1536xf32>)
  AIE.objectfifo.register_external_buffers @obj_out_flux (%t70, {%ext_buffer_out}) : (memref<512xf32>)

  func.func private @hdiff_lap_fp32(%AL: memref<256xf32>,%BL: memref<256xf32>, %CL:  memref<256xf32>, %DL: memref<256xf32>, %EL:  memref<256xf32>,  %OLL1: memref<256xf32>,  %OLL2: memref<256xf32>,  %OLL3: memref<256xf32>,  %OLL4: memref<256xf32>) -> ()
  
  %c13 = AIE.core(%t71) { 
    
    %lb = arith.constant 0 : index
    %ub = arith.constant 2: index
    %step = arith.constant 1 : index
    AIE.useLock(%lock71_14, "Acquire", 0) // start the timer
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectfifo.acquire @obj_in (Consume, 5) : !AIE.objectfifosubview<memref<256xf32>>
      %row0 = AIE.objectfifo.subview.access %obj_in_subview[0] : !AIE.objectfifosubview<memref<256xf32>> -> memref<256xf32>
      %row1 = AIE.objectfifo.subview.access %obj_in_subview[1] : !AIE.objectfifosubview<memref<256xf32>> -> memref<256xf32>
      %row2 = AIE.objectfifo.subview.access %obj_in_subview[2] : !AIE.objectfifosubview<memref<256xf32>> -> memref<256xf32>
      %row3 = AIE.objectfifo.subview.access %obj_in_subview[3] : !AIE.objectfifosubview<memref<256xf32>> -> memref<256xf32>
      %row4 = AIE.objectfifo.subview.access %obj_in_subview[4] : !AIE.objectfifosubview<memref<256xf32>> -> memref<256xf32>


      %obj_out_subview_lap = AIE.objectfifo.acquire @obj_out_lap (Produce, 4 ) : !AIE.objectfifosubview<memref<256xf32>>
      %obj_out_lap1 = AIE.objectfifo.subview.access %obj_out_subview_lap[0] : !AIE.objectfifosubview<memref<256xf32>> -> memref<256xf32>
      %obj_out_lap2 = AIE.objectfifo.subview.access %obj_out_subview_lap[1] : !AIE.objectfifosubview<memref<256xf32>> -> memref<256xf32>
      %obj_out_lap3 = AIE.objectfifo.subview.access %obj_out_subview_lap[2] : !AIE.objectfifosubview<memref<256xf32>> -> memref<256xf32>
      %obj_out_lap4 = AIE.objectfifo.subview.access %obj_out_subview_lap[3] : !AIE.objectfifosubview<memref<256xf32>> -> memref<256xf32>

      func.call @hdiff_lap_fp32(%row0,%row1,%row2,%row3,%row4,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4) : (memref<256xf32>,memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>,  memref<256xf32>,  memref<256xf32>,  memref<256xf32>,  memref<256xf32>) -> ()
      AIE.objectfifo.release @obj_out_lap (Produce, 4)
      AIE.objectfifo.release @obj_in (Consume, 1)

    }
    AIE.objectfifo.release @obj_in (Consume, 4)

    AIE.end
  } { link_with="hdiff_lap_fp32.o" }

func.func private @hdiff_flux_fp32(%AF: memref<256xf32>,%BF: memref<256xf32>, %CF:  memref<256xf32>,   %OLF1: memref<256xf32>,  %OLF2: memref<256xf32>,  %OLF3: memref<256xf32>,  %OLF4: memref<256xf32>,  %OF: memref<256xf32>) -> ()
  
  %c14 = AIE.core(%t72) { 
    
    %lb = arith.constant 0 : index
    %ub = arith.constant 2: index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectfifo.acquire @obj_in (Consume, 5) : !AIE.objectfifosubview<memref<256xf32>>
      
      %row1 = AIE.objectfifo.subview.access %obj_in_subview[1] : !AIE.objectfifosubview<memref<256xf32>> -> memref<256xf32>
      %row2 = AIE.objectfifo.subview.access %obj_in_subview[2] : !AIE.objectfifosubview<memref<256xf32>> -> memref<256xf32>
      %row3 = AIE.objectfifo.subview.access %obj_in_subview[3] : !AIE.objectfifosubview<memref<256xf32>> -> memref<256xf32>
      
      %obj_out_subview_lap = AIE.objectfifo.acquire @obj_out_lap (Consume, 4) : !AIE.objectfifosubview<memref<256xf32>>
      %obj_out_lap1 = AIE.objectfifo.subview.access %obj_out_subview_lap[0] : !AIE.objectfifosubview<memref<256xf32>> -> memref<256xf32>
      %obj_out_lap2 = AIE.objectfifo.subview.access %obj_out_subview_lap[1] : !AIE.objectfifosubview<memref<256xf32>> -> memref<256xf32>
      %obj_out_lap3 = AIE.objectfifo.subview.access %obj_out_subview_lap[2] : !AIE.objectfifosubview<memref<256xf32>> -> memref<256xf32>
      %obj_out_lap4 = AIE.objectfifo.subview.access %obj_out_subview_lap[3] : !AIE.objectfifosubview<memref<256xf32>> -> memref<256xf32>

      %obj_out_subview_flux = AIE.objectfifo.acquire @obj_out_flux (Produce, 1) : !AIE.objectfifosubview<memref<256xf32>>
      %obj_out_flux = AIE.objectfifo.subview.access %obj_out_subview_flux[0] : !AIE.objectfifosubview<memref<256xf32>> -> memref<256xf32>


      func.call @hdiff_flux_fp32(%row1,%row2,%row3,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4, %obj_out_flux ) : (memref<256xf32>,memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>,  memref<256xf32>,  memref<256xf32>) -> ()
      AIE.objectfifo.release @obj_out_lap (Consume, 4)
      AIE.objectfifo.release @obj_out_flux (Produce, 1)
      AIE.objectfifo.release @obj_in (Consume, 1)
    }
    AIE.useLock(%lock72_14, "Acquire", 0) // stop the timer
    AIE.objectfifo.release @obj_in (Consume, 4)

    AIE.end
  } { link_with="hdiff_flux_fp32.o" }



}

