// (c) 2023 SAFARI Research Group at ETH Zurich, Gagandeep Singh, D-ITET   
    
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


module @hdiff_bundle_4 {
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

//---Generating B-block 1---*-
//---col 0---*-
  %tile0_5 = AIE.tile(0, 5)
  %tile0_6 = AIE.tile(0, 6)
  %tile0_7 = AIE.tile(0, 7)
  %tile0_8 = AIE.tile(0, 8)
//---col 1---*-
  %tile1_5 = AIE.tile(1, 5)
  %tile1_6 = AIE.tile(1, 6)
  %tile1_7 = AIE.tile(1, 7)
  %tile1_8 = AIE.tile(1, 8)
//---col 2---*-
  %tile2_5 = AIE.tile(2, 5)
  %tile2_6 = AIE.tile(2, 6)
  %tile2_7 = AIE.tile(2, 7)
  %tile2_8 = AIE.tile(2, 8)

//---Generating B-block 2---*-
//---col 0---*-
  %tile3_1 = AIE.tile(3, 1)
  %tile3_2 = AIE.tile(3, 2)
  %tile3_3 = AIE.tile(3, 3)
  %tile3_4 = AIE.tile(3, 4)
//---col 1---*-
  %tile4_1 = AIE.tile(4, 1)
  %tile4_2 = AIE.tile(4, 2)
  %tile4_3 = AIE.tile(4, 3)
  %tile4_4 = AIE.tile(4, 4)
//---col 2---*-
  %tile5_1 = AIE.tile(5, 1)
  %tile5_2 = AIE.tile(5, 2)
  %tile5_3 = AIE.tile(5, 3)
  %tile5_4 = AIE.tile(5, 4)

//---Generating B-block 3---*-
//---col 0---*-
  %tile3_5 = AIE.tile(3, 5)
  %tile3_6 = AIE.tile(3, 6)
  %tile3_7 = AIE.tile(3, 7)
  %tile3_8 = AIE.tile(3, 8)
//---col 1---*-
  %tile4_5 = AIE.tile(4, 5)
  %tile4_6 = AIE.tile(4, 6)
  %tile4_7 = AIE.tile(4, 7)
  %tile4_8 = AIE.tile(4, 8)
//---col 2---*-
  %tile5_5 = AIE.tile(5, 5)
  %tile5_6 = AIE.tile(5, 6)
  %tile5_7 = AIE.tile(5, 7)
  %tile5_8 = AIE.tile(5, 8)

//---NOC Tile 2---*-
  %tile2_0 = AIE.tile(2, 0)
//---NOC Tile 3---*-
  %tile3_0 = AIE.tile(3, 0)

// timing locks
  %lock02_14 = AIE.lock(%tile0_2, 14) { sym_name = "lock02_14" }
  %lock22_14 = AIE.lock(%tile2_2, 14) { sym_name = "lock22_14" }

// timing locks
  %lock06_14 = AIE.lock(%tile0_6, 14) { sym_name = "lock06_14" }
  %lock26_14 = AIE.lock(%tile2_6, 14) { sym_name = "lock26_14" }

// timing locks
  %lock32_14 = AIE.lock(%tile3_2, 14) { sym_name = "lock32_14" }
  %lock52_14 = AIE.lock(%tile5_2, 14) { sym_name = "lock52_14" }

// timing locks
  %lock36_14 = AIE.lock(%tile3_6, 14) { sym_name = "lock36_14" }
  %lock56_14 = AIE.lock(%tile5_6, 14) { sym_name = "lock56_14" }

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
//---Generating B1 buffers---*-
  %block_1_buf_in_shim_2 = AIE.objectFifo.createObjectFifo(%tile2_0,{%tile0_5,%tile1_5,%tile0_6,%tile1_6,%tile0_7,%tile1_7,%tile0_8,%tile1_8},9) { sym_name = "block_1_buf_in_shim_2" } : !AIE.objectFifo<memref<256xi32>> //B block input
  %block_1_buf_row_5_inter_lap= AIE.objectFifo.createObjectFifo(%tile0_5,{%tile1_5},5){ sym_name ="block_1_buf_row_5_inter_lap"} : !AIE.objectFifo<memref<256xi32>>
  %block_1_buf_row_5_inter_flx1= AIE.objectFifo.createObjectFifo(%tile1_5,{%tile2_5},6) { sym_name ="block_1_buf_row_5_inter_flx1"} : !AIE.objectFifo<memref<512xi32>>
  %block_1_buf_row_5_out_flx2= AIE.objectFifo.createObjectFifo(%tile2_5,{%tile2_6},2) { sym_name ="block_1_buf_row_5_out_flx2"} : !AIE.objectFifo<memref<256xi32>>
  %block_1_buf_row_6_inter_lap= AIE.objectFifo.createObjectFifo(%tile0_6,{%tile1_6},5){ sym_name ="block_1_buf_row_6_inter_lap"} : !AIE.objectFifo<memref<256xi32>>
  %block_1_buf_row_6_inter_flx1= AIE.objectFifo.createObjectFifo(%tile1_6,{%tile2_6},6) { sym_name ="block_1_buf_row_6_inter_flx1"} : !AIE.objectFifo<memref<512xi32>>
  %block_1_buf_out_shim_2= AIE.objectFifo.createObjectFifo(%tile2_6,{%tile2_0},5){ sym_name ="block_1_buf_out_shim_2"} : !AIE.objectFifo<memref<256xi32>> //B block output
  %block_1_buf_row_7_inter_lap= AIE.objectFifo.createObjectFifo(%tile0_7,{%tile1_7},5){ sym_name ="block_1_buf_row_7_inter_lap"} : !AIE.objectFifo<memref<256xi32>>
  %block_1_buf_row_7_inter_flx1= AIE.objectFifo.createObjectFifo(%tile1_7,{%tile2_7},6) { sym_name ="block_1_buf_row_7_inter_flx1"} : !AIE.objectFifo<memref<512xi32>>
  %block_1_buf_row_7_out_flx2= AIE.objectFifo.createObjectFifo(%tile2_7,{%tile2_6},2) { sym_name ="block_1_buf_row_7_out_flx2"} : !AIE.objectFifo<memref<256xi32>>
  %block_1_buf_row_8_inter_lap= AIE.objectFifo.createObjectFifo(%tile0_8,{%tile1_8},5){ sym_name ="block_1_buf_row_8_inter_lap"} : !AIE.objectFifo<memref<256xi32>>
  %block_1_buf_row_8_inter_flx1= AIE.objectFifo.createObjectFifo(%tile1_8,{%tile2_8},6) { sym_name ="block_1_buf_row_8_inter_flx1"} : !AIE.objectFifo<memref<512xi32>>
  %block_1_buf_row_8_out_flx2= AIE.objectFifo.createObjectFifo(%tile2_8,{%tile2_6},2) { sym_name ="block_1_buf_row_8_out_flx2"} : !AIE.objectFifo<memref<256xi32>>
//---Generating B2 buffers---*-
  %block_2_buf_in_shim_3 = AIE.objectFifo.createObjectFifo(%tile3_0,{%tile3_1,%tile4_1,%tile3_2,%tile4_2,%tile3_3,%tile4_3,%tile3_4,%tile4_4},9) { sym_name = "block_2_buf_in_shim_3" } : !AIE.objectFifo<memref<256xi32>> //B block input
  %block_2_buf_row_1_inter_lap= AIE.objectFifo.createObjectFifo(%tile3_1,{%tile4_1},5){ sym_name ="block_2_buf_row_1_inter_lap"} : !AIE.objectFifo<memref<256xi32>>
  %block_2_buf_row_1_inter_flx1= AIE.objectFifo.createObjectFifo(%tile4_1,{%tile5_1},6) { sym_name ="block_2_buf_row_1_inter_flx1"} : !AIE.objectFifo<memref<512xi32>>
  %block_2_buf_row_1_out_flx2= AIE.objectFifo.createObjectFifo(%tile5_1,{%tile5_2},2) { sym_name ="block_2_buf_row_1_out_flx2"} : !AIE.objectFifo<memref<256xi32>>
  %block_2_buf_row_2_inter_lap= AIE.objectFifo.createObjectFifo(%tile3_2,{%tile4_2},5){ sym_name ="block_2_buf_row_2_inter_lap"} : !AIE.objectFifo<memref<256xi32>>
  %block_2_buf_row_2_inter_flx1= AIE.objectFifo.createObjectFifo(%tile4_2,{%tile5_2},6) { sym_name ="block_2_buf_row_2_inter_flx1"} : !AIE.objectFifo<memref<512xi32>>
  %block_2_buf_out_shim_3= AIE.objectFifo.createObjectFifo(%tile5_2,{%tile3_0},5){ sym_name ="block_2_buf_out_shim_3"} : !AIE.objectFifo<memref<256xi32>> //B block output
  %block_2_buf_row_3_inter_lap= AIE.objectFifo.createObjectFifo(%tile3_3,{%tile4_3},5){ sym_name ="block_2_buf_row_3_inter_lap"} : !AIE.objectFifo<memref<256xi32>>
  %block_2_buf_row_3_inter_flx1= AIE.objectFifo.createObjectFifo(%tile4_3,{%tile5_3},6) { sym_name ="block_2_buf_row_3_inter_flx1"} : !AIE.objectFifo<memref<512xi32>>
  %block_2_buf_row_3_out_flx2= AIE.objectFifo.createObjectFifo(%tile5_3,{%tile5_2},2) { sym_name ="block_2_buf_row_3_out_flx2"} : !AIE.objectFifo<memref<256xi32>>
  %block_2_buf_row_4_inter_lap= AIE.objectFifo.createObjectFifo(%tile3_4,{%tile4_4},5){ sym_name ="block_2_buf_row_4_inter_lap"} : !AIE.objectFifo<memref<256xi32>>
  %block_2_buf_row_4_inter_flx1= AIE.objectFifo.createObjectFifo(%tile4_4,{%tile5_4},6) { sym_name ="block_2_buf_row_4_inter_flx1"} : !AIE.objectFifo<memref<512xi32>>
  %block_2_buf_row_4_out_flx2= AIE.objectFifo.createObjectFifo(%tile5_4,{%tile5_2},2) { sym_name ="block_2_buf_row_4_out_flx2"} : !AIE.objectFifo<memref<256xi32>>
//---Generating B3 buffers---*-
  %block_3_buf_in_shim_3 = AIE.objectFifo.createObjectFifo(%tile3_0,{%tile3_5,%tile4_5,%tile3_6,%tile4_6,%tile3_7,%tile4_7,%tile3_8,%tile4_8},9) { sym_name = "block_3_buf_in_shim_3" } : !AIE.objectFifo<memref<256xi32>> //B block input
  %block_3_buf_row_5_inter_lap= AIE.objectFifo.createObjectFifo(%tile3_5,{%tile4_5},5){ sym_name ="block_3_buf_row_5_inter_lap"} : !AIE.objectFifo<memref<256xi32>>
  %block_3_buf_row_5_inter_flx1= AIE.objectFifo.createObjectFifo(%tile4_5,{%tile5_5},6) { sym_name ="block_3_buf_row_5_inter_flx1"} : !AIE.objectFifo<memref<512xi32>>
  %block_3_buf_row_5_out_flx2= AIE.objectFifo.createObjectFifo(%tile5_5,{%tile5_6},2) { sym_name ="block_3_buf_row_5_out_flx2"} : !AIE.objectFifo<memref<256xi32>>
  %block_3_buf_row_6_inter_lap= AIE.objectFifo.createObjectFifo(%tile3_6,{%tile4_6},5){ sym_name ="block_3_buf_row_6_inter_lap"} : !AIE.objectFifo<memref<256xi32>>
  %block_3_buf_row_6_inter_flx1= AIE.objectFifo.createObjectFifo(%tile4_6,{%tile5_6},6) { sym_name ="block_3_buf_row_6_inter_flx1"} : !AIE.objectFifo<memref<512xi32>>
  %block_3_buf_out_shim_3= AIE.objectFifo.createObjectFifo(%tile5_6,{%tile3_0},5){ sym_name ="block_3_buf_out_shim_3"} : !AIE.objectFifo<memref<256xi32>> //B block output
  %block_3_buf_row_7_inter_lap= AIE.objectFifo.createObjectFifo(%tile3_7,{%tile4_7},5){ sym_name ="block_3_buf_row_7_inter_lap"} : !AIE.objectFifo<memref<256xi32>>
  %block_3_buf_row_7_inter_flx1= AIE.objectFifo.createObjectFifo(%tile4_7,{%tile5_7},6) { sym_name ="block_3_buf_row_7_inter_flx1"} : !AIE.objectFifo<memref<512xi32>>
  %block_3_buf_row_7_out_flx2= AIE.objectFifo.createObjectFifo(%tile5_7,{%tile5_6},2) { sym_name ="block_3_buf_row_7_out_flx2"} : !AIE.objectFifo<memref<256xi32>>
  %block_3_buf_row_8_inter_lap= AIE.objectFifo.createObjectFifo(%tile3_8,{%tile4_8},5){ sym_name ="block_3_buf_row_8_inter_lap"} : !AIE.objectFifo<memref<256xi32>>
  %block_3_buf_row_8_inter_flx1= AIE.objectFifo.createObjectFifo(%tile4_8,{%tile5_8},6) { sym_name ="block_3_buf_row_8_inter_flx1"} : !AIE.objectFifo<memref<512xi32>>
  %block_3_buf_row_8_out_flx2= AIE.objectFifo.createObjectFifo(%tile5_8,{%tile5_6},2) { sym_name ="block_3_buf_row_8_out_flx2"} : !AIE.objectFifo<memref<256xi32>>
  %ext_buffer_in_0 = AIE.external_buffer  {sym_name = "ddr_buffer_in_0"}: memref<2304 x i32>
  %ext_buffer_out_0 = AIE.external_buffer  {sym_name = "ddr_buffer_out_0"}: memref<2048 x i32>

  %ext_buffer_in_1 = AIE.external_buffer  {sym_name = "ddr_buffer_in_1"}: memref<2304 x i32>
  %ext_buffer_out_1 = AIE.external_buffer  {sym_name = "ddr_buffer_out_1"}: memref<2048 x i32>

  %ext_buffer_in_2 = AIE.external_buffer  {sym_name = "ddr_buffer_in_2"}: memref<2304 x i32>
  %ext_buffer_out_2 = AIE.external_buffer  {sym_name = "ddr_buffer_out_2"}: memref<2048 x i32>

  %ext_buffer_in_3 = AIE.external_buffer  {sym_name = "ddr_buffer_in_3"}: memref<2304 x i32>
  %ext_buffer_out_3 = AIE.external_buffer  {sym_name = "ddr_buffer_out_3"}: memref<2048 x i32>

//Registering buffers
  AIE.objectFifo.registerExternalBuffers(%tile2_0, %block_0_buf_in_shim_2 : !AIE.objectFifo<memref<256xi32>>, {%ext_buffer_in_0}) : (memref<2304xi32>)
  AIE.objectFifo.registerExternalBuffers(%tile2_0, %block_0_buf_out_shim_2 : !AIE.objectFifo<memref<256xi32>>, {%ext_buffer_out_0}) : (memref<2048xi32>)

  AIE.objectFifo.registerExternalBuffers(%tile2_0, %block_1_buf_in_shim_2 : !AIE.objectFifo<memref<256xi32>>, {%ext_buffer_in_1}) : (memref<2304xi32>)
  AIE.objectFifo.registerExternalBuffers(%tile2_0, %block_1_buf_out_shim_2 : !AIE.objectFifo<memref<256xi32>>, {%ext_buffer_out_1}) : (memref<2048xi32>)

  AIE.objectFifo.registerExternalBuffers(%tile3_0, %block_2_buf_in_shim_3 : !AIE.objectFifo<memref<256xi32>>, {%ext_buffer_in_2}) : (memref<2304xi32>)
  AIE.objectFifo.registerExternalBuffers(%tile3_0, %block_2_buf_out_shim_3 : !AIE.objectFifo<memref<256xi32>>, {%ext_buffer_out_2}) : (memref<2048xi32>)

  AIE.objectFifo.registerExternalBuffers(%tile3_0, %block_3_buf_in_shim_3 : !AIE.objectFifo<memref<256xi32>>, {%ext_buffer_in_3}) : (memref<2304xi32>)
  AIE.objectFifo.registerExternalBuffers(%tile3_0, %block_3_buf_out_shim_3 : !AIE.objectFifo<memref<256xi32>>, {%ext_buffer_out_3}) : (memref<2048xi32>)


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
      AIE.objectFifo.release<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.objectFifo.release<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 7)
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
    AIE.useLock(%lock02_14, "Acquire", 0) // start the timer
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
      AIE.objectFifo.release<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.objectFifo.release<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 7)
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
      AIE.objectFifo.release<Consume>(%block_0_buf_row_1_out_flx2:!AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Consume>(%block_0_buf_row_3_out_flx2:!AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Consume>(%block_0_buf_row_4_out_flx2:!AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Produce>(%block_0_buf_out_shim_2:!AIE.objectFifo<memref<256xi32>>, 4)
    }
    AIE.useLock(%lock22_14, "Acquire", 0) // stop the timer
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
      AIE.objectFifo.release<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.objectFifo.release<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 7)
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
      AIE.objectFifo.release<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.objectFifo.release<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 7)
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

  %block_1_core0_5 = AIE.core(%tile0_5) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row0 = AIE.objectFifo.subview.access %obj_in_subview[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Produce>(%block_1_buf_row_5_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_lap(%row0,%row1,%row2,%row3,%row4,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Produce>(%block_1_buf_row_5_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
    }
    AIE.objectFifo.release<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 4)
    AIE.end
  } { link_with="hdiff_lap.o" }

  %block_1_core1_5 = AIE.core(%tile1_5) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Consume>(%block_1_buf_row_5_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux1 = AIE.objectFifo.acquire<Produce>(%block_1_buf_row_5_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_out_flux_inter1 = AIE.objectFifo.subview.access %obj_out_subview_flux1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter2 = AIE.objectFifo.subview.access %obj_out_subview_flux1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter3 = AIE.objectFifo.subview.access %obj_out_subview_flux1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter4 = AIE.objectFifo.subview.access %obj_out_subview_flux1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter5 = AIE.objectFifo.subview.access %obj_out_subview_flux1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      func.call @hdiff_flux1(%row1,%row2,%row3,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4, %obj_out_flux_inter1 , %obj_out_flux_inter2, %obj_out_flux_inter3, %obj_out_flux_inter4, %obj_out_flux_inter5) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_1_buf_row_5_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
      AIE.objectFifo.release<Produce>(%block_1_buf_row_5_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.objectFifo.release<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 7)
    AIE.end
  } { link_with="hdiff_flux1.o" }

  %block_1_core2_5 = AIE.core(%tile2_5) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_out_subview_flux_inter1 = AIE.objectFifo.acquire<Consume>(%block_1_buf_row_5_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_flux_inter_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element2 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element3 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element4 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element5 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      %obj_out_subview_flux = AIE.objectFifo.acquire<Produce>(%block_1_buf_row_5_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_flux_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_flux2(%obj_flux_inter_element1, %obj_flux_inter_element2,%obj_flux_inter_element3, %obj_flux_inter_element4, %obj_flux_inter_element5,  %obj_out_flux_element1 ) : ( memref<512xi32>,  memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_1_buf_row_5_inter_flx1 :!AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Produce>(%block_1_buf_row_5_out_flx2 :!AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.end
  } { link_with="hdiff_flux2.o" }

  %block_1_core0_6 = AIE.core(%tile0_6) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    AIE.useLock(%lock06_14, "Acquire", 0) // start the timer
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row0 = AIE.objectFifo.subview.access %obj_in_subview[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectFifo.subview.access %obj_in_subview[5] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Produce>(%block_1_buf_row_6_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_lap(%row0,%row1,%row2,%row3,%row4,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Produce>(%block_1_buf_row_6_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
    }
    AIE.objectFifo.release<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 4)
    AIE.end
  } { link_with="hdiff_lap.o" }

  %block_1_core1_6 = AIE.core(%tile1_6) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Consume>(%block_1_buf_row_6_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux1 = AIE.objectFifo.acquire<Produce>(%block_1_buf_row_6_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_out_flux_inter1 = AIE.objectFifo.subview.access %obj_out_subview_flux1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter2 = AIE.objectFifo.subview.access %obj_out_subview_flux1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter3 = AIE.objectFifo.subview.access %obj_out_subview_flux1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter4 = AIE.objectFifo.subview.access %obj_out_subview_flux1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter5 = AIE.objectFifo.subview.access %obj_out_subview_flux1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      func.call @hdiff_flux1(%row1,%row2,%row3,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4, %obj_out_flux_inter1 , %obj_out_flux_inter2, %obj_out_flux_inter3, %obj_out_flux_inter4, %obj_out_flux_inter5) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_1_buf_row_6_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
      AIE.objectFifo.release<Produce>(%block_1_buf_row_6_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.objectFifo.release<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 7)
    AIE.end
  } { link_with="hdiff_flux1.o" }

  // Gathering Tile
  %block_1_core2_6 = AIE.core(%tile2_6) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_out_subview_flux_inter1 = AIE.objectFifo.acquire<Consume>(%block_1_buf_row_6_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_flux_inter_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element2 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element3 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element4 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element5 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      %obj_out_subview_flux = AIE.objectFifo.acquire<Produce>(%block_1_buf_out_shim_2: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
  // Acquire all elements and add in order
      %obj_out_flux_element0 = AIE.objectFifo.subview.access %obj_out_subview_flux[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_flux_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_flux_element2 = AIE.objectFifo.subview.access %obj_out_subview_flux[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_flux_element3 = AIE.objectFifo.subview.access %obj_out_subview_flux[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
  // Acquiring outputs from other flux
      %obj_out_subview_flux1 = AIE.objectFifo.acquire<Consume>(%block_1_buf_row_5_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %final_out_from1 = AIE.objectFifo.subview.access %obj_out_subview_flux1[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux3 = AIE.objectFifo.acquire<Consume>(%block_1_buf_row_7_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %final_out_from3 = AIE.objectFifo.subview.access %obj_out_subview_flux3[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux4 = AIE.objectFifo.acquire<Consume>(%block_1_buf_row_8_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %final_out_from4 = AIE.objectFifo.subview.access %obj_out_subview_flux4[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

  // Ordering and copying data to gather tile (src-->dst)
      memref.copy %final_out_from1 , %obj_out_flux_element0 : memref<256xi32> to memref<256xi32>
      memref.copy %final_out_from3 , %obj_out_flux_element2 : memref<256xi32> to memref<256xi32>
      memref.copy %final_out_from4 , %obj_out_flux_element3 : memref<256xi32> to memref<256xi32>
      func.call @hdiff_flux2(%obj_flux_inter_element1, %obj_flux_inter_element2,%obj_flux_inter_element3, %obj_flux_inter_element4, %obj_flux_inter_element5,  %obj_out_flux_element1 ) : ( memref<512xi32>,  memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_1_buf_row_6_inter_flx1 :!AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Consume>(%block_1_buf_row_5_out_flx2:!AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Consume>(%block_1_buf_row_7_out_flx2:!AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Consume>(%block_1_buf_row_8_out_flx2:!AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Produce>(%block_1_buf_out_shim_2:!AIE.objectFifo<memref<256xi32>>, 4)
    }
    AIE.useLock(%lock26_14, "Acquire", 0) // stop the timer
    AIE.end
  } { link_with="hdiff_flux2.o" }

  %block_1_core0_7 = AIE.core(%tile0_7) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row0 = AIE.objectFifo.subview.access %obj_in_subview[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[5] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectFifo.subview.access %obj_in_subview[6] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Produce>(%block_1_buf_row_7_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_lap(%row0,%row1,%row2,%row3,%row4,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Produce>(%block_1_buf_row_7_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
    }
    AIE.objectFifo.release<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 4)
    AIE.end
  } { link_with="hdiff_lap.o" }

  %block_1_core1_7 = AIE.core(%tile1_7) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[5] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Consume>(%block_1_buf_row_7_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux1 = AIE.objectFifo.acquire<Produce>(%block_1_buf_row_7_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_out_flux_inter1 = AIE.objectFifo.subview.access %obj_out_subview_flux1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter2 = AIE.objectFifo.subview.access %obj_out_subview_flux1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter3 = AIE.objectFifo.subview.access %obj_out_subview_flux1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter4 = AIE.objectFifo.subview.access %obj_out_subview_flux1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter5 = AIE.objectFifo.subview.access %obj_out_subview_flux1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      func.call @hdiff_flux1(%row1,%row2,%row3,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4, %obj_out_flux_inter1 , %obj_out_flux_inter2, %obj_out_flux_inter3, %obj_out_flux_inter4, %obj_out_flux_inter5) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_1_buf_row_7_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
      AIE.objectFifo.release<Produce>(%block_1_buf_row_7_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.objectFifo.release<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 7)
    AIE.end
  } { link_with="hdiff_flux1.o" }

  %block_1_core2_7 = AIE.core(%tile2_7) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_out_subview_flux_inter1 = AIE.objectFifo.acquire<Consume>(%block_1_buf_row_7_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_flux_inter_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element2 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element3 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element4 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element5 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      %obj_out_subview_flux = AIE.objectFifo.acquire<Produce>(%block_1_buf_row_7_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_flux_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_flux2(%obj_flux_inter_element1, %obj_flux_inter_element2,%obj_flux_inter_element3, %obj_flux_inter_element4, %obj_flux_inter_element5,  %obj_out_flux_element1 ) : ( memref<512xi32>,  memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_1_buf_row_7_inter_flx1 :!AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Produce>(%block_1_buf_row_7_out_flx2 :!AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.end
  } { link_with="hdiff_flux2.o" }

  %block_1_core0_8 = AIE.core(%tile0_8) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row0 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[5] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[6] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectFifo.subview.access %obj_in_subview[7] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Produce>(%block_1_buf_row_8_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_lap(%row0,%row1,%row2,%row3,%row4,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Produce>(%block_1_buf_row_8_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
    }
    AIE.objectFifo.release<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 4)
    AIE.end
  } { link_with="hdiff_lap.o" }

  %block_1_core1_8 = AIE.core(%tile1_8) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[5] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[6] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Consume>(%block_1_buf_row_8_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux1 = AIE.objectFifo.acquire<Produce>(%block_1_buf_row_8_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_out_flux_inter1 = AIE.objectFifo.subview.access %obj_out_subview_flux1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter2 = AIE.objectFifo.subview.access %obj_out_subview_flux1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter3 = AIE.objectFifo.subview.access %obj_out_subview_flux1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter4 = AIE.objectFifo.subview.access %obj_out_subview_flux1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter5 = AIE.objectFifo.subview.access %obj_out_subview_flux1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      func.call @hdiff_flux1(%row1,%row2,%row3,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4, %obj_out_flux_inter1 , %obj_out_flux_inter2, %obj_out_flux_inter3, %obj_out_flux_inter4, %obj_out_flux_inter5) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_1_buf_row_8_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
      AIE.objectFifo.release<Produce>(%block_1_buf_row_8_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.objectFifo.release<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 7)
    AIE.end
  } { link_with="hdiff_flux1.o" }

  %block_1_core2_8 = AIE.core(%tile2_8) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_out_subview_flux_inter1 = AIE.objectFifo.acquire<Consume>(%block_1_buf_row_8_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_flux_inter_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element2 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element3 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element4 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element5 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      %obj_out_subview_flux = AIE.objectFifo.acquire<Produce>(%block_1_buf_row_8_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_flux_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_flux2(%obj_flux_inter_element1, %obj_flux_inter_element2,%obj_flux_inter_element3, %obj_flux_inter_element4, %obj_flux_inter_element5,  %obj_out_flux_element1 ) : ( memref<512xi32>,  memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_1_buf_row_8_inter_flx1 :!AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Produce>(%block_1_buf_row_8_out_flx2 :!AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.end
  } { link_with="hdiff_flux2.o" }

  %block_2_core3_1 = AIE.core(%tile3_1) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row0 = AIE.objectFifo.subview.access %obj_in_subview[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Produce>(%block_2_buf_row_1_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_lap(%row0,%row1,%row2,%row3,%row4,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Produce>(%block_2_buf_row_1_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
    }
    AIE.objectFifo.release<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 4)
    AIE.end
  } { link_with="hdiff_lap.o" }

  %block_2_core4_1 = AIE.core(%tile4_1) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Consume>(%block_2_buf_row_1_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux1 = AIE.objectFifo.acquire<Produce>(%block_2_buf_row_1_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_out_flux_inter1 = AIE.objectFifo.subview.access %obj_out_subview_flux1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter2 = AIE.objectFifo.subview.access %obj_out_subview_flux1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter3 = AIE.objectFifo.subview.access %obj_out_subview_flux1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter4 = AIE.objectFifo.subview.access %obj_out_subview_flux1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter5 = AIE.objectFifo.subview.access %obj_out_subview_flux1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      func.call @hdiff_flux1(%row1,%row2,%row3,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4, %obj_out_flux_inter1 , %obj_out_flux_inter2, %obj_out_flux_inter3, %obj_out_flux_inter4, %obj_out_flux_inter5) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_2_buf_row_1_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
      AIE.objectFifo.release<Produce>(%block_2_buf_row_1_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.objectFifo.release<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 7)
    AIE.end
  } { link_with="hdiff_flux1.o" }

  %block_2_core5_1 = AIE.core(%tile5_1) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_out_subview_flux_inter1 = AIE.objectFifo.acquire<Consume>(%block_2_buf_row_1_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_flux_inter_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element2 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element3 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element4 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element5 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      %obj_out_subview_flux = AIE.objectFifo.acquire<Produce>(%block_2_buf_row_1_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_flux_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_flux2(%obj_flux_inter_element1, %obj_flux_inter_element2,%obj_flux_inter_element3, %obj_flux_inter_element4, %obj_flux_inter_element5,  %obj_out_flux_element1 ) : ( memref<512xi32>,  memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_2_buf_row_1_inter_flx1 :!AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Produce>(%block_2_buf_row_1_out_flx2 :!AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.end
  } { link_with="hdiff_flux2.o" }

  %block_2_core3_2 = AIE.core(%tile3_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    AIE.useLock(%lock32_14, "Acquire", 0) // start the timer
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row0 = AIE.objectFifo.subview.access %obj_in_subview[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectFifo.subview.access %obj_in_subview[5] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Produce>(%block_2_buf_row_2_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_lap(%row0,%row1,%row2,%row3,%row4,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Produce>(%block_2_buf_row_2_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
    }
    AIE.objectFifo.release<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 4)
    AIE.end
  } { link_with="hdiff_lap.o" }

  %block_2_core4_2 = AIE.core(%tile4_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Consume>(%block_2_buf_row_2_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux1 = AIE.objectFifo.acquire<Produce>(%block_2_buf_row_2_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_out_flux_inter1 = AIE.objectFifo.subview.access %obj_out_subview_flux1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter2 = AIE.objectFifo.subview.access %obj_out_subview_flux1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter3 = AIE.objectFifo.subview.access %obj_out_subview_flux1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter4 = AIE.objectFifo.subview.access %obj_out_subview_flux1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter5 = AIE.objectFifo.subview.access %obj_out_subview_flux1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      func.call @hdiff_flux1(%row1,%row2,%row3,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4, %obj_out_flux_inter1 , %obj_out_flux_inter2, %obj_out_flux_inter3, %obj_out_flux_inter4, %obj_out_flux_inter5) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_2_buf_row_2_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
      AIE.objectFifo.release<Produce>(%block_2_buf_row_2_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.objectFifo.release<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 7)
    AIE.end
  } { link_with="hdiff_flux1.o" }

  // Gathering Tile
  %block_2_core5_2 = AIE.core(%tile5_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_out_subview_flux_inter1 = AIE.objectFifo.acquire<Consume>(%block_2_buf_row_2_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_flux_inter_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element2 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element3 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element4 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element5 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      %obj_out_subview_flux = AIE.objectFifo.acquire<Produce>(%block_2_buf_out_shim_3: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
  // Acquire all elements and add in order
      %obj_out_flux_element0 = AIE.objectFifo.subview.access %obj_out_subview_flux[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_flux_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_flux_element2 = AIE.objectFifo.subview.access %obj_out_subview_flux[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_flux_element3 = AIE.objectFifo.subview.access %obj_out_subview_flux[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
  // Acquiring outputs from other flux
      %obj_out_subview_flux1 = AIE.objectFifo.acquire<Consume>(%block_2_buf_row_1_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %final_out_from1 = AIE.objectFifo.subview.access %obj_out_subview_flux1[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux3 = AIE.objectFifo.acquire<Consume>(%block_2_buf_row_3_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %final_out_from3 = AIE.objectFifo.subview.access %obj_out_subview_flux3[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux4 = AIE.objectFifo.acquire<Consume>(%block_2_buf_row_4_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %final_out_from4 = AIE.objectFifo.subview.access %obj_out_subview_flux4[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

  // Ordering and copying data to gather tile (src-->dst)
      memref.copy %final_out_from1 , %obj_out_flux_element0 : memref<256xi32> to memref<256xi32>
      memref.copy %final_out_from3 , %obj_out_flux_element2 : memref<256xi32> to memref<256xi32>
      memref.copy %final_out_from4 , %obj_out_flux_element3 : memref<256xi32> to memref<256xi32>
      func.call @hdiff_flux2(%obj_flux_inter_element1, %obj_flux_inter_element2,%obj_flux_inter_element3, %obj_flux_inter_element4, %obj_flux_inter_element5,  %obj_out_flux_element1 ) : ( memref<512xi32>,  memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_2_buf_row_2_inter_flx1 :!AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Consume>(%block_2_buf_row_1_out_flx2:!AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Consume>(%block_2_buf_row_3_out_flx2:!AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Consume>(%block_2_buf_row_4_out_flx2:!AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Produce>(%block_2_buf_out_shim_3:!AIE.objectFifo<memref<256xi32>>, 4)
    }
    AIE.useLock(%lock52_14, "Acquire", 0) // stop the timer
    AIE.end
  } { link_with="hdiff_flux2.o" }

  %block_2_core3_3 = AIE.core(%tile3_3) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row0 = AIE.objectFifo.subview.access %obj_in_subview[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[5] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectFifo.subview.access %obj_in_subview[6] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Produce>(%block_2_buf_row_3_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_lap(%row0,%row1,%row2,%row3,%row4,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Produce>(%block_2_buf_row_3_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
    }
    AIE.objectFifo.release<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 4)
    AIE.end
  } { link_with="hdiff_lap.o" }

  %block_2_core4_3 = AIE.core(%tile4_3) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[5] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Consume>(%block_2_buf_row_3_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux1 = AIE.objectFifo.acquire<Produce>(%block_2_buf_row_3_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_out_flux_inter1 = AIE.objectFifo.subview.access %obj_out_subview_flux1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter2 = AIE.objectFifo.subview.access %obj_out_subview_flux1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter3 = AIE.objectFifo.subview.access %obj_out_subview_flux1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter4 = AIE.objectFifo.subview.access %obj_out_subview_flux1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter5 = AIE.objectFifo.subview.access %obj_out_subview_flux1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      func.call @hdiff_flux1(%row1,%row2,%row3,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4, %obj_out_flux_inter1 , %obj_out_flux_inter2, %obj_out_flux_inter3, %obj_out_flux_inter4, %obj_out_flux_inter5) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_2_buf_row_3_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
      AIE.objectFifo.release<Produce>(%block_2_buf_row_3_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.objectFifo.release<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 7)
    AIE.end
  } { link_with="hdiff_flux1.o" }

  %block_2_core5_3 = AIE.core(%tile5_3) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_out_subview_flux_inter1 = AIE.objectFifo.acquire<Consume>(%block_2_buf_row_3_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_flux_inter_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element2 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element3 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element4 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element5 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      %obj_out_subview_flux = AIE.objectFifo.acquire<Produce>(%block_2_buf_row_3_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_flux_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_flux2(%obj_flux_inter_element1, %obj_flux_inter_element2,%obj_flux_inter_element3, %obj_flux_inter_element4, %obj_flux_inter_element5,  %obj_out_flux_element1 ) : ( memref<512xi32>,  memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_2_buf_row_3_inter_flx1 :!AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Produce>(%block_2_buf_row_3_out_flx2 :!AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.end
  } { link_with="hdiff_flux2.o" }

  %block_2_core3_4 = AIE.core(%tile3_4) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row0 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[5] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[6] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectFifo.subview.access %obj_in_subview[7] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Produce>(%block_2_buf_row_4_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_lap(%row0,%row1,%row2,%row3,%row4,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Produce>(%block_2_buf_row_4_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
    }
    AIE.objectFifo.release<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 4)
    AIE.end
  } { link_with="hdiff_lap.o" }

  %block_2_core4_4 = AIE.core(%tile4_4) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[5] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[6] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Consume>(%block_2_buf_row_4_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux1 = AIE.objectFifo.acquire<Produce>(%block_2_buf_row_4_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_out_flux_inter1 = AIE.objectFifo.subview.access %obj_out_subview_flux1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter2 = AIE.objectFifo.subview.access %obj_out_subview_flux1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter3 = AIE.objectFifo.subview.access %obj_out_subview_flux1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter4 = AIE.objectFifo.subview.access %obj_out_subview_flux1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter5 = AIE.objectFifo.subview.access %obj_out_subview_flux1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      func.call @hdiff_flux1(%row1,%row2,%row3,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4, %obj_out_flux_inter1 , %obj_out_flux_inter2, %obj_out_flux_inter3, %obj_out_flux_inter4, %obj_out_flux_inter5) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_2_buf_row_4_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
      AIE.objectFifo.release<Produce>(%block_2_buf_row_4_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.objectFifo.release<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 7)
    AIE.end
  } { link_with="hdiff_flux1.o" }

  %block_2_core5_4 = AIE.core(%tile5_4) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_out_subview_flux_inter1 = AIE.objectFifo.acquire<Consume>(%block_2_buf_row_4_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_flux_inter_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element2 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element3 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element4 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element5 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      %obj_out_subview_flux = AIE.objectFifo.acquire<Produce>(%block_2_buf_row_4_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_flux_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_flux2(%obj_flux_inter_element1, %obj_flux_inter_element2,%obj_flux_inter_element3, %obj_flux_inter_element4, %obj_flux_inter_element5,  %obj_out_flux_element1 ) : ( memref<512xi32>,  memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_2_buf_row_4_inter_flx1 :!AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Produce>(%block_2_buf_row_4_out_flx2 :!AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.end
  } { link_with="hdiff_flux2.o" }

  %block_3_core3_5 = AIE.core(%tile3_5) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row0 = AIE.objectFifo.subview.access %obj_in_subview[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Produce>(%block_3_buf_row_5_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_lap(%row0,%row1,%row2,%row3,%row4,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Produce>(%block_3_buf_row_5_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
    }
    AIE.objectFifo.release<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 4)
    AIE.end
  } { link_with="hdiff_lap.o" }

  %block_3_core4_5 = AIE.core(%tile4_5) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Consume>(%block_3_buf_row_5_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux1 = AIE.objectFifo.acquire<Produce>(%block_3_buf_row_5_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_out_flux_inter1 = AIE.objectFifo.subview.access %obj_out_subview_flux1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter2 = AIE.objectFifo.subview.access %obj_out_subview_flux1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter3 = AIE.objectFifo.subview.access %obj_out_subview_flux1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter4 = AIE.objectFifo.subview.access %obj_out_subview_flux1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter5 = AIE.objectFifo.subview.access %obj_out_subview_flux1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      func.call @hdiff_flux1(%row1,%row2,%row3,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4, %obj_out_flux_inter1 , %obj_out_flux_inter2, %obj_out_flux_inter3, %obj_out_flux_inter4, %obj_out_flux_inter5) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_3_buf_row_5_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
      AIE.objectFifo.release<Produce>(%block_3_buf_row_5_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.objectFifo.release<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 7)
    AIE.end
  } { link_with="hdiff_flux1.o" }

  %block_3_core5_5 = AIE.core(%tile5_5) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_out_subview_flux_inter1 = AIE.objectFifo.acquire<Consume>(%block_3_buf_row_5_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_flux_inter_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element2 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element3 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element4 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element5 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      %obj_out_subview_flux = AIE.objectFifo.acquire<Produce>(%block_3_buf_row_5_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_flux_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_flux2(%obj_flux_inter_element1, %obj_flux_inter_element2,%obj_flux_inter_element3, %obj_flux_inter_element4, %obj_flux_inter_element5,  %obj_out_flux_element1 ) : ( memref<512xi32>,  memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_3_buf_row_5_inter_flx1 :!AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Produce>(%block_3_buf_row_5_out_flx2 :!AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.end
  } { link_with="hdiff_flux2.o" }

  %block_3_core3_6 = AIE.core(%tile3_6) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    AIE.useLock(%lock36_14, "Acquire", 0) // start the timer
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row0 = AIE.objectFifo.subview.access %obj_in_subview[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectFifo.subview.access %obj_in_subview[5] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Produce>(%block_3_buf_row_6_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_lap(%row0,%row1,%row2,%row3,%row4,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Produce>(%block_3_buf_row_6_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
    }
    AIE.objectFifo.release<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 4)
    AIE.end
  } { link_with="hdiff_lap.o" }

  %block_3_core4_6 = AIE.core(%tile4_6) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Consume>(%block_3_buf_row_6_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux1 = AIE.objectFifo.acquire<Produce>(%block_3_buf_row_6_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_out_flux_inter1 = AIE.objectFifo.subview.access %obj_out_subview_flux1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter2 = AIE.objectFifo.subview.access %obj_out_subview_flux1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter3 = AIE.objectFifo.subview.access %obj_out_subview_flux1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter4 = AIE.objectFifo.subview.access %obj_out_subview_flux1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter5 = AIE.objectFifo.subview.access %obj_out_subview_flux1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      func.call @hdiff_flux1(%row1,%row2,%row3,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4, %obj_out_flux_inter1 , %obj_out_flux_inter2, %obj_out_flux_inter3, %obj_out_flux_inter4, %obj_out_flux_inter5) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_3_buf_row_6_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
      AIE.objectFifo.release<Produce>(%block_3_buf_row_6_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.objectFifo.release<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 7)
    AIE.end
  } { link_with="hdiff_flux1.o" }

  // Gathering Tile
  %block_3_core5_6 = AIE.core(%tile5_6) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_out_subview_flux_inter1 = AIE.objectFifo.acquire<Consume>(%block_3_buf_row_6_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_flux_inter_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element2 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element3 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element4 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element5 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      %obj_out_subview_flux = AIE.objectFifo.acquire<Produce>(%block_3_buf_out_shim_3: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
  // Acquire all elements and add in order
      %obj_out_flux_element0 = AIE.objectFifo.subview.access %obj_out_subview_flux[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_flux_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_flux_element2 = AIE.objectFifo.subview.access %obj_out_subview_flux[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_flux_element3 = AIE.objectFifo.subview.access %obj_out_subview_flux[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
  // Acquiring outputs from other flux
      %obj_out_subview_flux1 = AIE.objectFifo.acquire<Consume>(%block_3_buf_row_5_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %final_out_from1 = AIE.objectFifo.subview.access %obj_out_subview_flux1[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux3 = AIE.objectFifo.acquire<Consume>(%block_3_buf_row_7_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %final_out_from3 = AIE.objectFifo.subview.access %obj_out_subview_flux3[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux4 = AIE.objectFifo.acquire<Consume>(%block_3_buf_row_8_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %final_out_from4 = AIE.objectFifo.subview.access %obj_out_subview_flux4[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

  // Ordering and copying data to gather tile (src-->dst)
      memref.copy %final_out_from1 , %obj_out_flux_element0 : memref<256xi32> to memref<256xi32>
      memref.copy %final_out_from3 , %obj_out_flux_element2 : memref<256xi32> to memref<256xi32>
      memref.copy %final_out_from4 , %obj_out_flux_element3 : memref<256xi32> to memref<256xi32>
      func.call @hdiff_flux2(%obj_flux_inter_element1, %obj_flux_inter_element2,%obj_flux_inter_element3, %obj_flux_inter_element4, %obj_flux_inter_element5,  %obj_out_flux_element1 ) : ( memref<512xi32>,  memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_3_buf_row_6_inter_flx1 :!AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Consume>(%block_3_buf_row_5_out_flx2:!AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Consume>(%block_3_buf_row_7_out_flx2:!AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Consume>(%block_3_buf_row_8_out_flx2:!AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Produce>(%block_3_buf_out_shim_3:!AIE.objectFifo<memref<256xi32>>, 4)
    }
    AIE.useLock(%lock56_14, "Acquire", 0) // stop the timer
    AIE.end
  } { link_with="hdiff_flux2.o" }

  %block_3_core3_7 = AIE.core(%tile3_7) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row0 = AIE.objectFifo.subview.access %obj_in_subview[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[5] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectFifo.subview.access %obj_in_subview[6] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Produce>(%block_3_buf_row_7_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_lap(%row0,%row1,%row2,%row3,%row4,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Produce>(%block_3_buf_row_7_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
    }
    AIE.objectFifo.release<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 4)
    AIE.end
  } { link_with="hdiff_lap.o" }

  %block_3_core4_7 = AIE.core(%tile4_7) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[5] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Consume>(%block_3_buf_row_7_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux1 = AIE.objectFifo.acquire<Produce>(%block_3_buf_row_7_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_out_flux_inter1 = AIE.objectFifo.subview.access %obj_out_subview_flux1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter2 = AIE.objectFifo.subview.access %obj_out_subview_flux1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter3 = AIE.objectFifo.subview.access %obj_out_subview_flux1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter4 = AIE.objectFifo.subview.access %obj_out_subview_flux1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter5 = AIE.objectFifo.subview.access %obj_out_subview_flux1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      func.call @hdiff_flux1(%row1,%row2,%row3,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4, %obj_out_flux_inter1 , %obj_out_flux_inter2, %obj_out_flux_inter3, %obj_out_flux_inter4, %obj_out_flux_inter5) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_3_buf_row_7_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
      AIE.objectFifo.release<Produce>(%block_3_buf_row_7_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.objectFifo.release<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 7)
    AIE.end
  } { link_with="hdiff_flux1.o" }

  %block_3_core5_7 = AIE.core(%tile5_7) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_out_subview_flux_inter1 = AIE.objectFifo.acquire<Consume>(%block_3_buf_row_7_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_flux_inter_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element2 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element3 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element4 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element5 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      %obj_out_subview_flux = AIE.objectFifo.acquire<Produce>(%block_3_buf_row_7_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_flux_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_flux2(%obj_flux_inter_element1, %obj_flux_inter_element2,%obj_flux_inter_element3, %obj_flux_inter_element4, %obj_flux_inter_element5,  %obj_out_flux_element1 ) : ( memref<512xi32>,  memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_3_buf_row_7_inter_flx1 :!AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Produce>(%block_3_buf_row_7_out_flx2 :!AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.end
  } { link_with="hdiff_flux2.o" }

  %block_3_core3_8 = AIE.core(%tile3_8) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row0 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[5] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[6] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectFifo.subview.access %obj_in_subview[7] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Produce>(%block_3_buf_row_8_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_lap(%row0,%row1,%row2,%row3,%row4,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Produce>(%block_3_buf_row_8_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
    }
    AIE.objectFifo.release<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 4)
    AIE.end
  } { link_with="hdiff_lap.o" }

  %block_3_core4_8 = AIE.core(%tile4_8) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[5] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[6] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Consume>(%block_3_buf_row_8_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux1 = AIE.objectFifo.acquire<Produce>(%block_3_buf_row_8_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_out_flux_inter1 = AIE.objectFifo.subview.access %obj_out_subview_flux1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter2 = AIE.objectFifo.subview.access %obj_out_subview_flux1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter3 = AIE.objectFifo.subview.access %obj_out_subview_flux1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter4 = AIE.objectFifo.subview.access %obj_out_subview_flux1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter5 = AIE.objectFifo.subview.access %obj_out_subview_flux1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      func.call @hdiff_flux1(%row1,%row2,%row3,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4, %obj_out_flux_inter1 , %obj_out_flux_inter2, %obj_out_flux_inter3, %obj_out_flux_inter4, %obj_out_flux_inter5) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_3_buf_row_8_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
      AIE.objectFifo.release<Produce>(%block_3_buf_row_8_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.objectFifo.release<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 7)
    AIE.end
  } { link_with="hdiff_flux1.o" }

  %block_3_core5_8 = AIE.core(%tile5_8) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_out_subview_flux_inter1 = AIE.objectFifo.acquire<Consume>(%block_3_buf_row_8_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_flux_inter_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element2 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element3 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element4 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element5 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      %obj_out_subview_flux = AIE.objectFifo.acquire<Produce>(%block_3_buf_row_8_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_flux_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_flux2(%obj_flux_inter_element1, %obj_flux_inter_element2,%obj_flux_inter_element3, %obj_flux_inter_element4, %obj_flux_inter_element5,  %obj_out_flux_element1 ) : ( memref<512xi32>,  memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_3_buf_row_8_inter_flx1 :!AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Produce>(%block_3_buf_row_8_out_flx2 :!AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.end
  } { link_with="hdiff_flux2.o" }

}
