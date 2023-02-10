// (c) 2023 SAFARI Research Group at ETH Zurich, Gagandeep Singh, D-ITET   
    
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


#include "test_library.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <xaiengine.h>
#include <time.h>  
#define HIGH_ADDR(addr) ((addr & 0xffffffff00000000) >> 32)
#define LOW_ADDR(addr) (addr & 0x00000000ffffffff)
#define MLIR_STACK_OFFSET 4096
#define B_BLOCK_DEPTH 4 //set how many rows
#define HDIFF_COL 3 //columns
#define START_ROW 1
#define INPUT_ROWS 9   
    
    #define TOTAL_B_BLOCK 2
 #include "aie_inc.cpp"

int main(int argc, char *argv[]) {
  printf("test start.");
  clock_t t; 

  aie_libxaie_ctx_t *_xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(_xaie);

  u32 sleep_u = 100000;
  usleep(sleep_u);
  printf("before configure cores.");
   for (int b=0; b<TOTAL_B_BLOCK;b++)
  {
    for (int i=0; i<HDIFF_COL;i++){
      for (int j=START_ROW; j<START_ROW+B_BLOCK_DEPTH;j++)
        mlir_aie_clear_tile_memory(_xaie, i, j);
    }
  }

//   mlir_aie_clear_tile_memory(_xaie, 6, 4);
  mlir_aie_configure_cores(_xaie);

  usleep(sleep_u);
  printf("before configure switchboxes.");
  mlir_aie_configure_switchboxes(_xaie);
  mlir_aie_initialize_locks(_xaie);

      mlir_aie_acquire_lock(_xaie, 0, 2, 14, 0, 0); // for timing
   XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(0,2), XAIE_MEM_MOD, 2,XAIE_EVENT_LOCK_14_ACQ_MEM);
   EventMonitor pc0(_xaie, 2, 2, 0, XAIE_EVENT_BROADCAST_2_MEM,XAIE_EVENT_LOCK_14_ACQ_MEM, XAIE_EVENT_NONE_MEM,XAIE_MEM_MOD);
   pc0.set();
 
  mlir_aie_acquire_lock(_xaie, 0, 6, 14, 0, 0); // for timing
   XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(0,6), XAIE_MEM_MOD, 2,XAIE_EVENT_LOCK_14_ACQ_MEM);
   EventMonitor pc1(_xaie, 2, 6, 0, XAIE_EVENT_BROADCAST_2_MEM,XAIE_EVENT_LOCK_14_ACQ_MEM, XAIE_EVENT_NONE_MEM,XAIE_MEM_MOD);
   pc1.set();
 

    usleep(sleep_u);
  printf("before configure DMA");
  mlir_aie_configure_dmas(_xaie);
  int errors = 0;
      mlir_aie_init_mems(_xaie, 4);
 
    printf("Finish configure");
  #define DMA_COUNT_IN 256*INPUT_ROWS
  #define DMA_COUNT_OUT 256*2*B_BLOCK_DEPTH

      int *ddr_ptr_in_0 = mlir_aie_mem_alloc(_xaie, 0, DMA_COUNT_IN);
   int *ddr_ptr_in_1 = mlir_aie_mem_alloc(_xaie, 1, DMA_COUNT_IN);
   int *ddr_ptr_out_0 = mlir_aie_mem_alloc(_xaie, 2, DMA_COUNT_OUT);
   int *ddr_ptr_out_1 = mlir_aie_mem_alloc(_xaie, 3, DMA_COUNT_OUT);
   for (int i = 0; i < DMA_COUNT_IN; i++) {
     *(ddr_ptr_in_0+ i) = i;
     *(ddr_ptr_in_1+ i) = i;
   }
   for (int i = 0; i < DMA_COUNT_OUT; i++) {
     *(ddr_ptr_out_0+ i) = i;
     *(ddr_ptr_out_1+ i) = i;
   }
   mlir_aie_sync_mem_dev(_xaie, 0);
   mlir_aie_sync_mem_dev(_xaie, 1);
   mlir_aie_sync_mem_dev(_xaie, 2);
   mlir_aie_sync_mem_dev(_xaie, 3);
 #ifdef LIBXAIENGINEV2
    mlir_aie_external_set_addr_ddr_buffer_in_0((u64)ddr_ptr_in_0); 
     mlir_aie_external_set_addr_ddr_buffer_in_1((u64)ddr_ptr_in_1); 
     mlir_aie_external_set_addr_ddr_buffer_out_0((u64)ddr_ptr_out_0); 
     mlir_aie_external_set_addr_ddr_buffer_out_1((u64)ddr_ptr_out_1); 
     mlir_aie_configure_shimdma_20(_xaie);
 #endif

    printf("before core start");
  // mlir_aie_print_tile_status(_xaie, 7, 3);

  printf("Release lock for accessing DDR.");
  mlir_aie_release_of_0_lock_0(_xaie, 1, 0); // (_xaie,release_value,time_out)
  mlir_aie_release_of_23_lock_0(_xaie, 1, 0); // (_xaie,release_value,time_out)
  mlir_aie_release_of_15_lock_0(_xaie, 0, 0);
  mlir_aie_release_of_38_lock_0(_xaie, 0, 0);



/*ADDD ALL THE LOCKS*/




  printf("Start cores");
  ///// --- start counter-----
  t = clock(); 
  mlir_aie_start_cores(_xaie);
      mlir_aie_release_lock(_xaie, 0, 2, 14, 0, 0); // for timing
   mlir_aie_release_lock(_xaie, 0, 6, 14, 0, 0); // for timing
 

         t = clock() - t; 

  printf ("It took %ld clicks (%f seconds).",t,((float)t)/CLOCKS_PER_SEC);

  usleep(sleep_u);
  printf("after core start");
  // mlir_aie_print_tile_status(_xaie, 7, 3);

  usleep(sleep_u);
  mlir_aie_sync_mem_cpu(_xaie, 2); //// only used in libaiev2 //sync up with output
   mlir_aie_sync_mem_cpu(_xaie, 3); //// only used in libaiev2 //sync up with output
 

      for (int i =0; i < 512; i ++ ){
    printf("Location %d:  %d", i, ddr_ptr_out_0[i]);
  }

  int res = 0;
  if (!errors) {
    printf("PASS!");
    res = 0;
  } else {
    printf("Fail!");
    res = -1;
  } 
  printf("PC0 cycles: %d", pc0.diff());
   printf("PC1 cycles: %d", pc1.diff());
 

  mlir_aie_deinit_libxaie(_xaie);

  printf("test done.");

  return res;
}

    
    
    