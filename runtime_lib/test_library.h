//===- test_library.h -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include <xaiengine.h>
#include <stdio.h>

#define ACDC_check(s, r, v, errors) if(r != v) {printf("ERROR %s: %s expected %d, but was %d!\n", s, #r, v, r); errors++;}
#define ACDC_check_float(s, r, v, errors) if(r != v) {printf("ERROR %s: %s expected %f, but was %f!\n", s, #r, v, r); errors++;}

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

/// Zero out the configuration memory of the shim tile.
void ACDC_clear_shim_config(struct XAieGbl_Tile &tile);