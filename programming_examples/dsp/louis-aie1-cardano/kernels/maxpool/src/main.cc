//===- main.cc -----------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
#include <string.h>
#include <stdio.h>

#ifndef __chess_
#define __PTHREAD_API__
#define __NEW_X86Sim__
#endif

#include "cardano/window/window.h"
#include "cardano/stream/streams.h"
#include <cardano/redefine.h>
#include <cardano/intrinsics.h>

#include "maxpool.h"
#include "test_params.h"

#define S_AIn 0
#define S_AOutRef 1

alignas(128) uint8_t AIn[inTileSize];
alignas(128) uint8_t AOut[outTileSize];

void fillBuff(uint8_t* buff, size_t size, int from) {
    //printf("size: %ld\n", size);
    for(size_t i = 0; i < size; i++) {
        buff[i] = get_ss(from);
        printf("%4d, ", buff[i]);
    }
    printf("\n");
}

int main() {
    printf("Testing M; %d, N; %d, C; %d, MP_W; %d, MP_S: %d\n", M, N, C, MP_W, MP_S);
    printf("Load A...\n");
    fillBuff(AIn, inTileSize, S_AIn);

    memset(AOut, '\0', outTileSize);

    uint8_t AOutRef[outTileSize];
    fillBuff(AOutRef, outTileSize, S_AOutRef);

    printf("Init Windows...\n");

    window_internal actInWindow;
    window_internal actOutWindow;

    size_t vecSizeInA = sizeof(AIn) / sizeof(v16uint8); // will load 128 at a time
    window_init(&actInWindow, 1, (v16uint8*)AIn, vecSizeInA, vecSizeInA);

    size_t vecSizeOutA = sizeof(AOut) / sizeof(v16uint8); // will load 128 at a time
    window_init(&actOutWindow, 1, (v16uint8*)AOut, vecSizeOutA, vecSizeOutA);

    printf("Compute..\n");
    maxpool_int8<M, N, C, MP_S, MP_W>((input_window_uint8*)&actInWindow,
                                      (output_window_uint8*)&actOutWindow);

    printf("Check...\n");
    for(size_t i = 0; i < outTileSize; i++) {
        printf("%4d, ", AOut[i]);
    }
    printf("\n");

    for(unsigned int i = 0; i < outTileSize; i++) {
        if(AOutRef[i] != AOut[i]) {
            const unsigned int x = (i/8) % outWidth;

            if(x < outWidthR) {
                printf("Failed at %d with ref: %d, computed: %d\n", i, AOutRef[i], AOut[i]);
                assert(AOutRef[i] == AOut[i]);
            }
        }
    }

    printf("SUCCESS!\n");

    return 0;
}
