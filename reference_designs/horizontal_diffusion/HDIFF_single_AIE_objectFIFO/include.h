//===- include.h ------------------------------------------------*- C++ -*-===//
//
// (c) 2023 SAFARI Research Group at ETH Zurich, Gagandeep Singh, D-ITET
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
//
//===----------------------------------------------------------------------===//

#pragma once
#define VECTORIZED_KERNEL
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