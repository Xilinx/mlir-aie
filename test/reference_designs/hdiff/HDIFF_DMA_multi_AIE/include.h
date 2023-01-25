/*  (c) Copyright 2014 - 2015 Xilinx, Inc. All rights reserved.

    This file contains confidential and proprietary information
    of Xilinx, Inc. and is protected under U.S. and
    international copyright and other intellectual property
    laws.

    DISCLAIMER
    This disclaimer is not a license and does not grant any
    rights to the materials distributed herewith. Except as
    otherwise provided in a valid license issued to you by
    Xilinx, and to the maximum extent permitted by applicable
    law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
    WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
    AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
    BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
    INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
    (2) Xilinx shall not be liable (whether in contract or tort,
    including negligence, or under any other theory of
    liability) for any loss or damage of any kind or nature
    related to, arising under or in connection with these
    materials, including for any direct, or any indirect,
    special, incidental, or consequential loss or damage
    (including loss of data, profits, goodwill, or any type of
    loss or damage suffered as a result of any action brought
    by a third party) even if such damage or loss was
    reasonably foreseeable or Xilinx had been advised of the
    possibility of the same.

    CRITICAL APPLICATIONS
    Xilinx products are not designed or intended to be fail-
    safe, or for use in any application requiring fail-safe
    performance, such as life-support or safety devices or
    systems, Class III medical devices, nuclear facilities,
    applications related to the deployment of airbags, or any
    other applications that could lead to death, personal
    injury, or severe property or environmental damage
    (individually and collectively, "Critical
    Applications"). Customer assumes the sole risk and
    liability of any use of Xilinx products in Critical
    Applications, subject only to applicable laws and
    regulations governing limitations on product liability.

    THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
    PART OF THIS FILE AT ALL TIMES.                       */


#pragma once
// #define WITH_MARGIN
#define VECTORIZED_KERNEL
// #define MULTI_CORE
// #define  MULTI_4x4
// #include <adf.h>
#include <stdint.h>
#define GRIDROW     256
#define GRIDCOL     256
#define GRIDDEPTH   1
#define TOTAL_INPUT GRIDROW*GRIDCOL*GRIDDEPTH

#define ROW			256
#define COL			256
#define TILE_SIZE  COL

#define WMARGIN     256       // Margin before the frame


#define NBYTES		4		// datatype byte-width

#define AVAIL_CORES 25*25

#define CORE_REQUIRED TOTAL_INPUT/TILE_SIZE

#ifdef MULTI_CORE
    #ifdef  MULTI_2x2
        #define HW_ROW 2
        #define HW_COL 2
    #else 
        #define HW_ROW 4
        #define HW_COL 4
    #endif

    #define USED_CORE HW_ROW*HW_COL
    #define NITER    TOTAL_INPUT/(USED_CORE*TILE_SIZE)   // Number of iteration
#else

#define NITER    TOTAL_INPUT/(TILE_SIZE)    // Number of iteration
#endif

#ifdef WITH_MARGIN
#define INPUT_FILE "./data/TestInputS.txt"    // Input file name and location
#else
#define INPUT_FILE "./data/dataset_256x256x64.txt"    // Input file name and location
#endif
#define OUTPUT_FILE "./data/TestOutputS.txt"    // Output file name and location