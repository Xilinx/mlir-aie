# /scratch/gagandee/mlir-air/utils/mlir-aie/reference_designs/horizontal_diffusion/HDIFF_tri_AIE_objectFIFO_ping_pong_scaled/gen_hdiff_cpp.py -*- Python -*-
#
# (c) 2023 SAFARI Research Group at ETH Zurich, Gagandeep Singh, D-ITET
#
# This file is licensed under the MIT License.
# SPDX-License-Identifier: MIT
#


noc_columns = [2, 3, 6, 7, 10, 11, 18, 19, 26, 27, 34, 35, 42, 43, 46, 47]

total_b_block = 8  # only 1
b_block_depth = 4  # set how many rows
input_rows = 9  # data input per block row

hdiff_col = 3  # columns
arraycols = 0  # must be even until 32
broadcast_cores = 0  # only 1-2
arrayrows = 0  # one for processing and one for shimDMA
startrow = 1
startcol = 0
bufsize = 256  # must fit in data memory
bufsize_flx1 = 512
bufsize_flx2 = 256
dram_bufsize_in = 256 * (input_rows)
dram_bufsize_out = 256 * 2 * b_block_depth

iter_i = 0
iter_j = 1
cur_noc_count = 0
cur_noc_count_in = 0
cur_noc_count_out = 0


def main():
    global arrayrows
    global arraycols
    global bufsize

    print(
        "Enabling %d block with depth %d = %d AIE cores"
        % (total_b_block, b_block_depth, total_b_block * b_block_depth * hdiff_col)
    )

    rows = arrayrows  # row 0 is reserved
    cols = arraycols

    f = open("test_%d.cpp" % (total_b_block), "w+")
    f.write(
        """//===- hdiff.cc -------------------------------------------------*- C++ -*-===//  
//  
// (c) 2023 SAFARI Research Group at ETH Zurich, Gagandeep Singh, D-ITET
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
// 
//
//===----------------------------------------------------------------------===//\n\n\n"""
    )

    def noc_div_two_channel(block):
        # print(block)
        seg = 2
        if block < seg:
            val = noc_columns[0]
        elif block >= seg and block < 2 * seg:
            val = noc_columns[1]
        elif block >= 2 * seg and block < 3 * seg:
            val = noc_columns[2]
        elif block >= 3 * seg and block < 4 * seg:
            val = noc_columns[3]
        elif block >= 4 * seg and block < 5 * seg:
            val = noc_columns[4]
        elif block >= 5 * seg and block < 6 * seg:
            val = noc_columns[5]
        elif block >= 6 * seg and block < 7 * seg:
            val = noc_columns[6]
        elif block >= 7 * seg and block < 8 * seg:
            val = noc_columns[7]
        elif block >= 8 * seg and block < 9 * seg:
            val = noc_columns[8]
        elif block >= 9 * seg and block < 10 * seg:
            val = noc_columns[9]
        elif block >= 10 * seg and block < 11 * seg:
            val = noc_columns[10]
        elif block >= 11 * seg and block < 12 * seg:
            val = noc_columns[11]
        elif block >= 12 * seg and block < 13 * seg:
            val = noc_columns[12]
        elif block >= 13 * seg and block < 14 * seg:
            val = noc_columns[13]
        elif block >= 14 * seg and block < 15 * seg:
            val = noc_columns[14]
        elif block >= 15 * seg:
            val = noc_columns[15]
        return val

    f.write(
        """#include "test_library.h"
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
"""
    )
    f.write("#define TOTAL_B_BLOCK %d\n" % (total_b_block))

    f.write(
        """#include "aie_inc.cpp"

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
"""
    )

    # setting timing locks
    b_col_shift = 0
    b_row_shift = 0
    count = 0
    for b in range(0, total_b_block):
        # f.write("// timing locks\n")
        if b % 2 == 0 and b != 0:
            b_col_shift = b_col_shift + 1
        for col in range(startcol, startcol + hdiff_col):  # col 0 is reserved in aie
            # we only have 16 unique broadcast events
            if count < 16:
                if col == 0:
                    f.write(
                        "  mlir_aie_acquire_lock(_xaie, %d, %d, 14, 0, 0); // for timing\n"
                        % (col + b_col_shift * 3, startrow + b_row_shift * 4)
                    )
                    f.write(
                        "  XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(%d,%d), XAIE_MEM_MOD, %d,XAIE_EVENT_LOCK_14_ACQ_MEM);\n"
                        % (col + b_col_shift * 3, startrow + b_row_shift * 4, count)
                    )
                if col == (startcol + hdiff_col - 1):
                    f.write(
                        "  EventMonitor pc%d(_xaie, %d, %d, 0, XAIE_EVENT_BROADCAST_%d_MEM,XAIE_EVENT_LOCK_14_ACQ_MEM, XAIE_EVENT_NONE_MEM,XAIE_MEM_MOD);\n"
                        % (
                            count,
                            col + b_col_shift * 3,
                            startrow + b_row_shift * 4,
                            count,
                        )
                    )
                    f.write("  pc%d.set();\n" % (count))
                    count = count + 1
        if b % 2 == 0:
            b_row_shift = b_row_shift + 1
        else:
            b_row_shift = 0
        # f.write("\n")

    f.write(
        """
  usleep(sleep_u);
  printf("before configure DMA");
  mlir_aie_configure_dmas(_xaie);
  int errors = 0;
"""
    )

    f.write("  mlir_aie_init_mems(_xaie, %d);\n" % (total_b_block * 2))

    f.write(
        """
  printf("Finish configure");
  #define DMA_COUNT_IN 256*INPUT_ROWS
  #define DMA_COUNT_OUT 256*2*B_BLOCK_DEPTH
"""
    )
    for i in range(0, total_b_block):  # col 0 is reserved in aie
        f.write(
            "  int *ddr_ptr_in_%d = mlir_aie_mem_alloc(_xaie, %d, DMA_COUNT_IN);\n"
            % (i, i)
        )

    for i in range(0, total_b_block):  # col 0 is reserved in aie
        f.write(
            "  int *ddr_ptr_out_%d = mlir_aie_mem_alloc(_xaie, %d, DMA_COUNT_OUT);\n"
            % (i, total_b_block + i)
        )

    f.write("  for (int i = 0; i < DMA_COUNT_IN; i++) {\n")
    for i in range(0, total_b_block):  # col 0 is reserved in aie
        f.write("    *(ddr_ptr_in_%d+ i) = i;\n" % (i))
    f.write("  }\n")

    f.write("  for (int i = 0; i < DMA_COUNT_OUT; i++) {\n")
    for i in range(0, total_b_block):  # col 0 is reserved in aie
        f.write("    *(ddr_ptr_out_%d+ i) = i;\n" % (i))
    f.write("  }\n")

    for i in range(0, total_b_block):  # col 0 is reserved in aie
        f.write("  mlir_aie_sync_mem_dev(_xaie, %d);\n" % (i))
    for i in range(0, total_b_block):  # col 0 is reserved in aie
        f.write("  mlir_aie_sync_mem_dev(_xaie, %d);\n" % (total_b_block + i))

    for i in range(0, total_b_block):  # col 0 is reserved in aie
        f.write(
            "    mlir_aie_external_set_addr_ddr_buffer_in_%d((u64)ddr_ptr_in_%d); \n"
            % (i, i)
        )

    for i in range(0, total_b_block):  # col 0 is reserved in aie
        f.write(
            "    mlir_aie_external_set_addr_ddr_buffer_out_%d((u64)ddr_ptr_out_%d); \n"
            % (i, i)
        )

    for block in range(0, total_b_block):  #
        if block % 2 == 0:
            shim_place = noc_div_two_channel(block)
            f.write("    mlir_aie_configure_shimdma_%d0(_xaie);\n" % (shim_place))
    f.write(
        """
  printf("before core start");
  // mlir_aie_print_tile_status(_xaie, 7, 3);

  printf("Release lock for accessing DDR.");
  mlir_aie_release_of_0_lock_0(_xaie, 1, 0); // (_xaie,release_value,time_out)
  mlir_aie_release_of_15_lock_0(_xaie, 0, 0);

/*ADDD ALL THE LOCKS*/

  printf("Start cores");
  ///// --- start counter-----
  t = clock(); 
  mlir_aie_start_cores(_xaie);
"""
    )

    # release timing locks
    b_col_shift = 0
    b_row_shift = 0
    count = 0
    broadcast_event = 0
    for b in range(0, total_b_block):
        # f.write("// timing locks\n")
        if b % 2 == 0 and b != 0:
            b_col_shift = b_col_shift + 1
        for col in range(startcol, startcol + hdiff_col):  # col 0 is reserved in aie
            if (col == 0) and (broadcast_event < 16):
                f.write(
                    "  mlir_aie_release_lock(_xaie, %d, %d, 14, 0, 0); // for timing\n"
                    % (col + b_col_shift * 3, startrow + b_row_shift * 4)
                )
                broadcast_event += 1
        if b % 2 == 0:
            b_row_shift = b_row_shift + 1
        else:
            b_row_shift = 0

    f.write(
        """

  t = clock() - t; 

  printf ("It took %ld clicks (%f seconds).",t,((float)t)/CLOCKS_PER_SEC);

  usleep(sleep_u);
  printf("after core start");
  // mlir_aie_print_tile_status(_xaie, 7, 3);

  usleep(sleep_u);
"""
    )
    for i in range(0, total_b_block):  # col 0 is reserved in aie
        f.write(
            "  mlir_aie_sync_mem_cpu(_xaie, %d); //// only used in libaiev2 //sync up with output\n"
            % (total_b_block + i)
        )

    f.write("\n")

    f.write(
        """
  for (int i =0; i < 512; i ++ ){
    printf("Location %d:  %d\\n", i, ddr_ptr_out_0[i]);
  }

  int res = 0;
  if (!errors) {
    printf("PASS!");
    res = 0;
  } else {
    printf("Fail!");
    res = -1;
  } """
    )

    f.write("\n")
    # setting timing locks
    b_col_shift = 0
    b_row_shift = 0
    count = 0
    for b in range(0, total_b_block):
        # f.write("// timing locks\n")
        if b % 2 == 0 and b != 0:
            b_col_shift = b_col_shift + 1
        for col in range(startcol, startcol + hdiff_col):  # col 0 is reserved in aie
            if (col == (startcol + hdiff_col - 1)) and count < 16:
                f.write(
                    '  printf("PC%d cycles: %%d\\n", pc%d.diff());\n' % (count, count)
                )
                count = count + 1
        if b % 2 == 0:
            b_row_shift = b_row_shift + 1
        else:
            b_row_shift = 0

    f.write(
        """

  mlir_aie_deinit_libxaie(_xaie);

  printf("test done.");

  return res;
}

    
    
    """
    )


if __name__ == "__main__":
    main()
