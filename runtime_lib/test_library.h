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

#define MODE_CORE 0
#define MODE_PL 1
#define MODE_MEM 2


#define MAP_SIZE 16UL
#define MAP_MASK (MAP_SIZE - 1)

#define ACDC_check(s, r, v, errors) if(r != v) {printf("ERROR %s: %s expected %d, but was %d!\n", s, #r, v, r); errors++;}
#define ACDC_check_float(s, r, v, errors) if(r != v) {printf("ERROR %s: %s expected %f, but was %f!\n", s, #r, v, r); errors++;}

#ifdef LIBXAIENGINEV2

/// Dump the contents of the memory associated with the given tile.
void ACDC_dump_tile_memory(XAie_DevInst *DevInst, XAie_LocType loc);

/// Clear the contents of the memory associated with the given tile.
void ACDC_clear_tile_memory(XAie_DevInst *DevInst, XAie_LocType loc);

/// Print the status of a dma represented by the given tile.
void ACDC_print_dma_status(XAie_DevInst *DevInst, XAie_LocType loc);

/// Print the status of a core represented by the given tile.
void ACDC_print_tile_status(XAie_DevInst *DevInst, XAie_LocType loc);

/// Zero out the program and configuration memory of the tile.
void ACDC_clear_config(XAie_DevInst *DevInst, XAie_LocType loc);

/// Zero out the configuration memory of the shim tile.
void ACDC_clear_shim_config(XAie_DevInst *DevInst, XAie_LocType loc);

#else
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
#endif

// class for using events and PF cpounters
class EventMon{
    public:
        EventMon(struct XAieGbl_Tile *_tilePtr, u32 _pfc, u32 _startE, u32 _endE, u32 _resetE, u8 _mode){
            tilePtr = _tilePtr;
            pfc = _pfc;
            mode = _mode; // 0: Core, 1: PL, 2, Mem
            if(mode == MODE_CORE){
                XAieTileCore_PerfCounterControl(tilePtr, pfc, _startE, _endE, _resetE);}
            else if(mode == MODE_PL){
                XAieTilePl_PerfCounterControl(tilePtr, pfc, _startE, _endE, _resetE);}
            else{
                XAieTileMem_PerfCounterControl(tilePtr, pfc, _startE, _endE, _resetE);}
            
        }
        void set(){
            if(mode == MODE_CORE){
                start = XAieTileCore_PerfCounterGet(tilePtr, pfc);}
            else if(mode == MODE_PL){
                start = XAieTilePl_PerfCounterGet(tilePtr, pfc);}
            else{
                start = XAieTileMem_PerfCounterGet(tilePtr, pfc);}
        }
        u32 read(){
            if(mode == MODE_CORE){
                return XAieTileCore_PerfCounterGet(tilePtr, pfc);}
            else if(mode == MODE_PL){
                return XAieTilePl_PerfCounterGet(tilePtr, pfc);}
            else{
                return XAieTileMem_PerfCounterGet(tilePtr, pfc);}
        }
        u32 diff(){
            u32 end;
            if(mode == MODE_CORE){
                end = XAieTileCore_PerfCounterGet(tilePtr, pfc);}
            else if(mode == MODE_PL){
                end = XAieTilePl_PerfCounterGet(tilePtr, pfc);}
            else{
                end = XAieTileMem_PerfCounterGet(tilePtr, pfc);}
            if(end < start){
                printf("WARNING: EventMon: performance counter wrapped!\n");
                return 0; // TODO: fix this
            }
            else{
                return end - start;
            }
        }
    private:
        u32 start;
        u32 pfc;
        u8 mode;
        struct XAieGbl_Tile *tilePtr;
};

void computeStats(u32 performance_counter[], int n);