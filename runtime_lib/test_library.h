// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

#include <xaiengine.h>
#include <stdio.h>

#define ACDC_check(s, r, v) if(r != v) {printf("ERROR %s: %s expected %d, but was %d!\n", s, #r, v, r); errors++;}
#define ACDC_check_float(s, r, v) if(r != v) {printf("ERROR %s: %s expected %f, but was %f!\n", s, #r, v, r); errors++;}

/// Dump the contents of the memory associated with the given tile.
void ACDC_dump_tile_memory(struct XAieGbl_Tile &tile);

/// Clear the contents of the memory associated with the given tile.
void ACDC_clear_tile_memory(struct XAieGbl_Tile &tile);

/// Print the status of a dma represented by the given tile.
void ACDC_print_dma_status(struct XAieGbl_Tile &tile);

/// Print the status of a core represented by the given tile.
void ACDC_print_tile_status(struct XAieGbl_Tile &tile);

/// Zero out the program and configuration memory of the tile.
void ACDC_clear_config(struct XAieGbl_Tile &tile);