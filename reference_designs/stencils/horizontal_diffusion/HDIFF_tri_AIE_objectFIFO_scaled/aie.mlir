// (c) 2023 SAFARI Research Group at ETH Zurich, Gagandeep Singh, D-ITET   
    
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


module @hdiff_bundle_1 {
//---Generating B-block 0---*-
//---col 0---*-
  %tile0_1 = AIE.tile(0, 1)
  %tile0_2 = AIE.tile(0, 2)
  %tile0_3 = AIE.tile(0, 3)
  %tile0_4 = AIE.tile(0, 4)
//---col 1---*-
  %tile1_1 = AIE.tile(1, 1)
  %tile1_2 = AIE.tile(1, 2)
  %tile1_3 = AIE.tile(1, 3)
  %tile1_4 = AIE.tile(1, 4)
//---col 2---*-
  %tile2_1 = AIE.tile(2, 1)
  %tile2_2 = AIE.tile(2, 2)
  %tile2_3 = AIE.tile(2, 3)
  %tile2_4 = AIE.tile(2, 4)


//---NOC Tile 2---*-
  %tile2_0 = AIE.tile(2, 0)

//---Generating B0 buffers---*-
  %block_0_buf_in_shim_2 = AIE.objectFifo.createObjectFifo(%tile2_0,{%tile0_1,%tile1_1,%tile0_2,%tile1_2,%tile0_3,%tile1_3,%tile0_4,%tile1_4},9) { sym_name = "block_0_buf_in_shim_2" } : !AIE.objectFifo<memref<256xi32>> //B block input
  %block_0_buf_row_1_inter_lap= AIE.objectFifo.createObjectFifo(%tile0_1,{%tile1_1},5){ sym_name ="block_0_buf_row_1_inter_lap"} : !AIE.objectFifo<memref<256xi32>>
  %block_0_buf_row_1_inter_flx1= AIE.objectFifo.createObjectFifo(%tile1_1,{%tile2_1},6) { sym_name ="block_0_buf_row_1_inter_flx1"} : !AIE.objectFifo<memref<512xi32>>
  %block_0_buf_row_1_out_flx2= AIE.objectFifo.createObjectFifo(%tile2_1,{%tile2_2},2) { sym_name ="block_0_buf_row_1_out_flx2"} : !AIE.objectFifo<memref<256xi32>>
  %block_0_buf_row_2_inter_lap= AIE.objectFifo.createObjectFifo(%tile0_2,{%tile1_2},5){ sym_name ="block_0_buf_row_2_inter_lap"} : !AIE.objectFifo<memref<256xi32>>
  %block_0_buf_row_2_inter_flx1= AIE.objectFifo.createObjectFifo(%tile1_2,{%tile2_2},6) { sym_name ="block_0_buf_row_2_inter_flx1"} : !AIE.objectFifo<memref<512xi32>>
  %block_0_buf_out_shim_2= AIE.objectFifo.createObjectFifo(%tile2_2,{%tile2_0},5){ sym_name ="block_0_buf_out_shim_2"} : !AIE.objectFifo<memref<256xi32>> //B block output
  %block_0_buf_row_3_inter_lap= AIE.objectFifo.createObjectFifo(%tile0_3,{%tile1_3},5){ sym_name ="block_0_buf_row_3_inter_lap"} : !AIE.objectFifo<memref<256xi32>>
  %block_0_buf_row_3_inter_flx1= AIE.objectFifo.createObjectFifo(%tile1_3,{%tile2_3},6) { sym_name ="block_0_buf_row_3_inter_flx1"} : !AIE.objectFifo<memref<512xi32>>
  %block_0_buf_row_3_out_flx2= AIE.objectFifo.createObjectFifo(%tile2_3,{%tile2_2},2) { sym_name ="block_0_buf_row_3_out_flx2"} : !AIE.objectFifo<memref<256xi32>>
  %block_0_buf_row_4_inter_lap= AIE.objectFifo.createObjectFifo(%tile0_4,{%tile1_4},5){ sym_name ="block_0_buf_row_4_inter_lap"} : !AIE.objectFifo<memref<256xi32>>
  %block_0_buf_row_4_inter_flx1= AIE.objectFifo.createObjectFifo(%tile1_4,{%tile2_4},6) { sym_name ="block_0_buf_row_4_inter_flx1"} : !AIE.objectFifo<memref<512xi32>>
  %block_0_buf_row_4_out_flx2= AIE.objectFifo.createObjectFifo(%tile2_4,{%tile2_2},2) { sym_name ="block_0_buf_row_4_out_flx2"} : !AIE.objectFifo<memref<256xi32>>
  %ext_buffer_in_0 = AIE.external_buffer  {sym_name = "ddr_buffer_in_0"}: memref<2048 x i32>
  %ext_buffer_out_0 = AIE.external_buffer  {sym_name = "ddr_buffer_out_0"}: memref<2048 x i32>

//Registering buffers
  AIE.objectFifo.registerExternalBuffers(%tile2_0, %block_0_buf_in_shim_2 : !AIE.objectFifo<memref<256xi32>>, {%ext_buffer_in_0}) : (memref<2048xi32>)
  AIE.objectFifo.registerExternalBuffers(%tile2_0, %block_0_buf_out_shim_2 : !AIE.objectFifo<memref<256xi32>>, {%ext_buffer_out_0}) : (memref<2048xi32>)


  func.func private @hdiff_lap(%AL: memref<256xi32>,%BL: memref<256xi32>, %CL:  memref<256xi32>, %DL: memref<256xi32>, %EL:  memref<256xi32>,  %OLL1: memref<256xi32>,  %OLL2: memref<256xi32>,  %OLL3: memref<256xi32>,  %OLL4: memref<256xi32>) -> ()
  func.func private @hdiff_flux1(%AF: memref<256xi32>,%BF: memref<256xi32>, %CF:  memref<256xi32>,   %OLF1: memref<256xi32>,  %OLF2: memref<256xi32>,  %OLF3: memref<256xi32>,  %OLF4: memref<256xi32>,  %OFI1: memref<512xi32>,  %OFI2: memref<512xi32>,  %OFI3: memref<512xi32>,  %OFI4: memref<512xi32>,  %OFI5: memref<512xi32>) -> ()
  func.func private @hdiff_flux2( %Inter1: memref<512xi32>,%Inter2: memref<512xi32>, %Inter3: memref<512xi32>,%Inter4: memref<512xi32>,%Inter5: memref<512xi32>,  %Out: memref<256xi32>) -> ()

  %block_0_core0_1 = AIE.core(%tile0_1) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row0 = AIE.objectFifo.subview.access %obj_in_subview[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Produce>(%block_0_buf_row_1_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_lap(%row0,%row1,%row2,%row3,%row4,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Produce>(%block_0_buf_row_1_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
    }
    AIE.objectFifo.release<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 4)
    AIE.end
  } { link_with="hdiff_lap.o" }

  %block_0_core1_1 = AIE.core(%tile1_1) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Consume>(%block_0_buf_row_1_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux1 = AIE.objectFifo.acquire<Produce>(%block_0_buf_row_1_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_out_flux_inter1 = AIE.objectFifo.subview.access %obj_out_subview_flux1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter2 = AIE.objectFifo.subview.access %obj_out_subview_flux1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter3 = AIE.objectFifo.subview.access %obj_out_subview_flux1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter4 = AIE.objectFifo.subview.access %obj_out_subview_flux1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter5 = AIE.objectFifo.subview.access %obj_out_subview_flux1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      func.call @hdiff_flux1(%row1,%row2,%row3,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4, %obj_out_flux_inter1 , %obj_out_flux_inter2, %obj_out_flux_inter3, %obj_out_flux_inter4, %obj_out_flux_inter5) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_0_buf_row_1_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
      AIE.objectFifo.release<Produce>(%block_0_buf_row_1_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5)
    }
    AIE.objectFifo.release<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 3)
    AIE.end
  } { link_with="hdiff_flux1.o" }

  %block_0_core2_1 = AIE.core(%tile2_1) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_out_subview_flux_inter1 = AIE.objectFifo.acquire<Consume>(%block_0_buf_row_1_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_flux_inter_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element2 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element3 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element4 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element5 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      %obj_out_subview_flux = AIE.objectFifo.acquire<Produce>(%block_0_buf_row_1_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_flux_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_flux2(%obj_flux_inter_element1, %obj_flux_inter_element2,%obj_flux_inter_element3, %obj_flux_inter_element4, %obj_flux_inter_element5,  %obj_out_flux_element1 ) : ( memref<512xi32>,  memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_0_buf_row_1_inter_flx1 :!AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Produce>(%block_0_buf_row_1_out_flx2 :!AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.end
  } { link_with="hdiff_flux2.o" }

  %block_0_core0_2 = AIE.core(%tile0_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row0 = AIE.objectFifo.subview.access %obj_in_subview[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectFifo.subview.access %obj_in_subview[5] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Produce>(%block_0_buf_row_2_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_lap(%row0,%row1,%row2,%row3,%row4,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Produce>(%block_0_buf_row_2_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
    }
    AIE.objectFifo.release<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 4)
    AIE.end
  } { link_with="hdiff_lap.o" }

  %block_0_core1_2 = AIE.core(%tile1_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Consume>(%block_0_buf_row_2_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux1 = AIE.objectFifo.acquire<Produce>(%block_0_buf_row_2_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_out_flux_inter1 = AIE.objectFifo.subview.access %obj_out_subview_flux1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter2 = AIE.objectFifo.subview.access %obj_out_subview_flux1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter3 = AIE.objectFifo.subview.access %obj_out_subview_flux1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter4 = AIE.objectFifo.subview.access %obj_out_subview_flux1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter5 = AIE.objectFifo.subview.access %obj_out_subview_flux1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      func.call @hdiff_flux1(%row1,%row2,%row3,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4, %obj_out_flux_inter1 , %obj_out_flux_inter2, %obj_out_flux_inter3, %obj_out_flux_inter4, %obj_out_flux_inter5) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_0_buf_row_2_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
      AIE.objectFifo.release<Produce>(%block_0_buf_row_2_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5)
    }
    AIE.objectFifo.release<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 3)
    AIE.end
  } { link_with="hdiff_flux1.o" }

  // Gathering Tile
  %block_0_core2_2 = AIE.core(%tile2_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_out_subview_flux_inter1 = AIE.objectFifo.acquire<Consume>(%block_0_buf_row_2_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_flux_inter_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element2 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element3 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element4 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element5 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      %obj_out_subview_flux = AIE.objectFifo.acquire<Produce>(%block_0_buf_out_shim_2: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
  // Acquire all elements and add in order
      %obj_out_flux_element0 = AIE.objectFifo.subview.access %obj_out_subview_flux[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_flux_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_flux_element2 = AIE.objectFifo.subview.access %obj_out_subview_flux[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_flux_element3 = AIE.objectFifo.subview.access %obj_out_subview_flux[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
  // Acquiring outputs from other flux
      %obj_out_subview_flux1 = AIE.objectFifo.acquire<Consume>(%block_0_buf_row_1_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %final_out_from1 = AIE.objectFifo.subview.access %obj_out_subview_flux1[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux3 = AIE.objectFifo.acquire<Consume>(%block_0_buf_row_3_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %final_out_from3 = AIE.objectFifo.subview.access %obj_out_subview_flux3[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux4 = AIE.objectFifo.acquire<Consume>(%block_0_buf_row_4_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %final_out_from4 = AIE.objectFifo.subview.access %obj_out_subview_flux4[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

  // Ordering and copying data to gather tile (src-->dst)
      memref.copy %final_out_from1 , %obj_out_flux_element0 : memref<256xi32> to memref<256xi32>
      memref.copy %final_out_from3 , %obj_out_flux_element2 : memref<256xi32> to memref<256xi32>
      memref.copy %final_out_from4 , %obj_out_flux_element3 : memref<256xi32> to memref<256xi32>
      func.call @hdiff_flux2(%obj_flux_inter_element1, %obj_flux_inter_element2,%obj_flux_inter_element3, %obj_flux_inter_element4, %obj_flux_inter_element5,  %obj_out_flux_element1 ) : ( memref<512xi32>,  memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_0_buf_row_2_inter_flx1 :!AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Produce>(%block_0_buf_out_shim_2 :!AIE.objectFifo<memref<256xi32>>, 4)
    }
    AIE.end
  } { link_with="hdiff_flux2.o" }

  %block_0_core0_3 = AIE.core(%tile0_3) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row0 = AIE.objectFifo.subview.access %obj_in_subview[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[5] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectFifo.subview.access %obj_in_subview[6] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Produce>(%block_0_buf_row_3_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_lap(%row0,%row1,%row2,%row3,%row4,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Produce>(%block_0_buf_row_3_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
    }
    AIE.objectFifo.release<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 4)
    AIE.end
  } { link_with="hdiff_lap.o" }

  %block_0_core1_3 = AIE.core(%tile1_3) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[5] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Consume>(%block_0_buf_row_3_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux1 = AIE.objectFifo.acquire<Produce>(%block_0_buf_row_3_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_out_flux_inter1 = AIE.objectFifo.subview.access %obj_out_subview_flux1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter2 = AIE.objectFifo.subview.access %obj_out_subview_flux1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter3 = AIE.objectFifo.subview.access %obj_out_subview_flux1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter4 = AIE.objectFifo.subview.access %obj_out_subview_flux1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter5 = AIE.objectFifo.subview.access %obj_out_subview_flux1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      func.call @hdiff_flux1(%row1,%row2,%row3,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4, %obj_out_flux_inter1 , %obj_out_flux_inter2, %obj_out_flux_inter3, %obj_out_flux_inter4, %obj_out_flux_inter5) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_0_buf_row_3_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
      AIE.objectFifo.release<Produce>(%block_0_buf_row_3_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5)
    }
    AIE.objectFifo.release<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 3)
    AIE.end
  } { link_with="hdiff_flux1.o" }

  %block_0_core2_3 = AIE.core(%tile2_3) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_out_subview_flux_inter1 = AIE.objectFifo.acquire<Consume>(%block_0_buf_row_3_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_flux_inter_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element2 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element3 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element4 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element5 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      %obj_out_subview_flux = AIE.objectFifo.acquire<Produce>(%block_0_buf_row_3_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_flux_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_flux2(%obj_flux_inter_element1, %obj_flux_inter_element2,%obj_flux_inter_element3, %obj_flux_inter_element4, %obj_flux_inter_element5,  %obj_out_flux_element1 ) : ( memref<512xi32>,  memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_0_buf_row_3_inter_flx1 :!AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Produce>(%block_0_buf_row_3_out_flx2 :!AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.end
  } { link_with="hdiff_flux2.o" }

  %block_0_core0_4 = AIE.core(%tile0_4) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row0 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[5] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[6] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectFifo.subview.access %obj_in_subview[7] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Produce>(%block_0_buf_row_4_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_lap(%row0,%row1,%row2,%row3,%row4,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Produce>(%block_0_buf_row_4_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
    }
    AIE.objectFifo.release<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 4)
    AIE.end
  } { link_with="hdiff_lap.o" }

  %block_0_core1_4 = AIE.core(%tile1_4) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[5] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[6] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Consume>(%block_0_buf_row_4_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux1 = AIE.objectFifo.acquire<Produce>(%block_0_buf_row_4_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_out_flux_inter1 = AIE.objectFifo.subview.access %obj_out_subview_flux1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter2 = AIE.objectFifo.subview.access %obj_out_subview_flux1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter3 = AIE.objectFifo.subview.access %obj_out_subview_flux1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter4 = AIE.objectFifo.subview.access %obj_out_subview_flux1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter5 = AIE.objectFifo.subview.access %obj_out_subview_flux1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      func.call @hdiff_flux1(%row1,%row2,%row3,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4, %obj_out_flux_inter1 , %obj_out_flux_inter2, %obj_out_flux_inter3, %obj_out_flux_inter4, %obj_out_flux_inter5) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_0_buf_row_4_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
      AIE.objectFifo.release<Produce>(%block_0_buf_row_4_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5)
    }
    AIE.objectFifo.release<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 3)
    AIE.end
  } { link_with="hdiff_flux1.o" }

  %block_0_core2_4 = AIE.core(%tile2_4) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_out_subview_flux_inter1 = AIE.objectFifo.acquire<Consume>(%block_0_buf_row_4_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_flux_inter_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element2 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element3 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element4 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element5 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      %obj_out_subview_flux = AIE.objectFifo.acquire<Produce>(%block_0_buf_row_4_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_flux_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_flux2(%obj_flux_inter_element1, %obj_flux_inter_element2,%obj_flux_inter_element3, %obj_flux_inter_element4, %obj_flux_inter_element5,  %obj_out_flux_element1 ) : ( memref<512xi32>,  memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_0_buf_row_4_inter_flx1 :!AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Produce>(%block_0_buf_row_4_out_flx2 :!AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.end
  } { link_with="hdiff_flux2.o" }

}
