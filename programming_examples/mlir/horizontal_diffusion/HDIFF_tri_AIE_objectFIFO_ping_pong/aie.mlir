//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// (c) 2023 SAFARI Research Group at ETH Zurich, Gagandeep Singh, D-ITET
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
// 
//
//===----------------------------------------------------------------------===//



module @hdiff_tri_AIE {
  %t73 = aie.tile(7, 3)
  %t72 = aie.tile(7, 2)
  %t71 = aie.tile(7, 1)
  %t70 = aie.tile(7, 0)
  
  %lock71_14 = aie.lock(%t71, 14) {sym_name = "lock71_14"}
  %lock73_14 = aie.lock(%t73, 14) {sym_name = "lock73_14"}

  aie.objectfifo @obj_in (%t70, {%t71, %t72}, 6 : i32) : !aie.objectfifo<memref<256xi32>>
  aie.objectfifo @obj_out_lap (%t71, {%t72}, 5 : i32)  : !aie.objectfifo<memref<256xi32>>
  aie.objectfifo @obj_out_flux_inter1 (%t72, {%t73}, 6 : i32) : !aie.objectfifo<memref<512xi32>>
  aie.objectfifo @obj_out_flux (%t73, {%t70}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
   
  // DDR buffer
  %ext_buffer_in0  = aie.external_buffer {sym_name = "ddr_test_buffer_in0"} : memref<1536 x i32>
  %ext_buffer_out = aie.external_buffer {sym_name = "ddr_test_buffer_out"} : memref<512 x i32>
      
  // Register the external memory pointers to the object FIFOs.
  aie.objectfifo.register_external_buffers @obj_in (%t70, {%ext_buffer_in0}) : (memref<1536xi32>)
  aie.objectfifo.register_external_buffers @obj_out_flux (%t70, {%ext_buffer_out}) : (memref<512xi32>)

  func.func private @hdiff_lap(%AL: memref<256xi32>,%BL: memref<256xi32>, %CL:  memref<256xi32>, %DL: memref<256xi32>, %EL:  memref<256xi32>,  %OLL1: memref<256xi32>,  %OLL2: memref<256xi32>,  %OLL3: memref<256xi32>,  %OLL4: memref<256xi32>) -> ()
  func.func private @hdiff_flux1(%AF: memref<256xi32>,%BF: memref<256xi32>, %CF:  memref<256xi32>,   %OLF1: memref<256xi32>,  %OLF2: memref<256xi32>,  %OLF3: memref<256xi32>,  %OLF4: memref<256xi32>,  %OFI1: memref<512xi32>,  %OFI2: memref<512xi32>,  %OFI3: memref<512xi32>,  %OFI4: memref<512xi32>,  %OFI5: memref<512xi32>) -> ()
  func.func private @hdiff_flux2( %Inter1: memref<512xi32>,%Inter2: memref<512xi32>, %Inter3: memref<512xi32>,%Inter4: memref<512xi32>,%Inter5: memref<512xi32>,  %Out: memref<256xi32>) -> ()

  %c13 = aie.core(%t71) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2: index
    %step = arith.constant 1 : index
    aie.use_lock(%lock71_14, "Acquire", 0) // start the timer
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = aie.objectfifo.acquire @obj_in (Consume, 5) : !aie.objectfifosubview<memref<256xi32>>
      %row0 = aie.objectfifo.subview.access %obj_in_subview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row1 = aie.objectfifo.subview.access %obj_in_subview[1] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = aie.objectfifo.subview.access %obj_in_subview[2] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = aie.objectfifo.subview.access %obj_in_subview[3] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row4 = aie.objectfifo.subview.access %obj_in_subview[4] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = aie.objectfifo.acquire @obj_out_lap (Produce, 4 ) : !aie.objectfifosubview<memref<256xi32>>
      %obj_out_lap1 = aie.objectfifo.subview.access %obj_out_subview_lap[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = aie.objectfifo.subview.access %obj_out_subview_lap[1] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = aie.objectfifo.subview.access %obj_out_subview_lap[2] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = aie.objectfifo.subview.access %obj_out_subview_lap[3] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_lap(%row0,%row1,%row2,%row3,%row4,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>) -> ()
      aie.objectfifo.release @obj_out_lap (Produce, 4)
      aie.objectfifo.release @obj_in (Consume, 1)
    }
    aie.objectfifo.release @obj_in (Consume, 4)

    aie.end
  } { link_with="hdiff_lap.o" }

  %c14 = aie.core(%t72) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2: index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = aie.objectfifo.acquire @obj_in (Consume, 5) : !aie.objectfifosubview<memref<256xi32>>

      %row1 = aie.objectfifo.subview.access %obj_in_subview[1] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row2 = aie.objectfifo.subview.access %obj_in_subview[2] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %row3 = aie.objectfifo.subview.access %obj_in_subview[3] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = aie.objectfifo.acquire @obj_out_lap (Consume, 4) : !aie.objectfifosubview<memref<256xi32>>
      %obj_out_lap1 = aie.objectfifo.subview.access %obj_out_subview_lap[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = aie.objectfifo.subview.access %obj_out_subview_lap[1] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = aie.objectfifo.subview.access %obj_out_subview_lap[2] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = aie.objectfifo.subview.access %obj_out_subview_lap[3] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux1 = aie.objectfifo.acquire @obj_out_flux_inter1 (Produce, 5) : !aie.objectfifosubview<memref<512xi32>>
      %obj_out_flux_inter1 = aie.objectfifo.subview.access %obj_out_subview_flux1[0] : !aie.objectfifosubview<memref<512xi32>> -> memref<512xi32>

      %obj_out_flux_inter2 = aie.objectfifo.subview.access %obj_out_subview_flux1[1] : !aie.objectfifosubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter3 = aie.objectfifo.subview.access %obj_out_subview_flux1[2] : !aie.objectfifosubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter4 = aie.objectfifo.subview.access %obj_out_subview_flux1[3] : !aie.objectfifosubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter5 = aie.objectfifo.subview.access %obj_out_subview_flux1[4] : !aie.objectfifosubview<memref<512xi32>> -> memref<512xi32>

      func.call @hdiff_flux1(%row1,%row2,%row3,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4, %obj_out_flux_inter1 , %obj_out_flux_inter2, %obj_out_flux_inter3, %obj_out_flux_inter4, %obj_out_flux_inter5) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>) -> ()
      aie.objectfifo.release @obj_out_lap (Consume, 4)
      aie.objectfifo.release @obj_out_flux_inter1 (Produce, 5)
      aie.objectfifo.release @obj_in (Consume, 1)
    }
    aie.objectfifo.release @obj_in (Consume, 4)

    aie.end
  } { link_with="hdiff_flux1.o" }

  %c15 = aie.core(%t73) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2: index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  

      %obj_out_subview_flux_inter1 = aie.objectfifo.acquire @obj_out_flux_inter1 (Consume, 5) : !aie.objectfifosubview<memref<512xi32>>
      %obj_flux_inter_element1 = aie.objectfifo.subview.access %obj_out_subview_flux_inter1[0] : !aie.objectfifosubview<memref<512xi32>> -> memref<512xi32>

      %obj_flux_inter_element2 = aie.objectfifo.subview.access %obj_out_subview_flux_inter1[1] : !aie.objectfifosubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element3 = aie.objectfifo.subview.access %obj_out_subview_flux_inter1[2] : !aie.objectfifosubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element4 = aie.objectfifo.subview.access %obj_out_subview_flux_inter1[3] : !aie.objectfifosubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element5 = aie.objectfifo.subview.access %obj_out_subview_flux_inter1[4] : !aie.objectfifosubview<memref<512xi32>> -> memref<512xi32>

      %obj_out_subview_flux = aie.objectfifo.acquire @obj_out_flux (Produce, 1) : !aie.objectfifosubview<memref<256xi32>>
      %obj_out_flux_element = aie.objectfifo.subview.access %obj_out_subview_flux[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_flux2(%obj_flux_inter_element1, %obj_flux_inter_element2,%obj_flux_inter_element3, %obj_flux_inter_element4, %obj_flux_inter_element5,  %obj_out_flux_element ) : ( memref<512xi32>,  memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<256xi32>) -> ()
      aie.objectfifo.release @obj_out_flux_inter1 (Consume, 5)
      aie.objectfifo.release @obj_out_flux (Produce, 1)
    }
    aie.use_lock(%lock73_14, "Acquire", 0) // stop the timer

    aie.end
  } { link_with="hdiff_flux2.o" }
}
