//===- generator.cc -----------------------------------000---*- C++ -*-===//
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
#include <stdlib.h>
#include <cassert>
#include <stdint.h>

#include "test_params.h"
#include "avg_params.h"

#define FNAME_ACTS "../../reference/activations.csv"
#define FNAME_MP_RES "../../reference/maxpool.csv"


void printActs(int* buf, size_t size) {
    for(size_t i = 0; i < size; i++) {
        if(i != 0 && (i%(8*MR)) == 0) {
            printf("\n\n");
        }

        printf("%4d,", buf[i]);
    }

    printf("\n\n");
}

void printMatToFile(int* buf, size_t size, const char* const filename) {
    FILE* f = fopen(filename, "w");
    for(size_t i = 0; i < size; i++) {
        fprintf(f, "%4d\n", buf[i]);
    }

    fflush(f);
    fclose(f);
}

void fillAct(int* buf, size_t size) {
    for(size_t i = 0; i < size; i++) {
        unsigned int c = i % 8 + (i/(8*MR*N)) * 8;
        unsigned int x = i/8 % MR;
        unsigned int y = i/(8*MR) % N;

        int value = (rand() % 256);

        if((c >= C) || (x >= M)) {
            buf[i] = 0;
        } else {
            buf[i] = (value % 256);
            //buf[i] = (((i)%(MR*N*8)) % 256);
            //if(buf[i] > 127) {
            //    buf[i] = -128 + (buf[i] - 128);
            //}
        }
    }
}

void readActsFromFile(int* buf, size_t size, const char* const filename) {
    FILE* f = fopen(filename, "r");

    for(size_t i = 0; i < size; i++) {
        unsigned int x = i % M;
        unsigned int y = (i / M) % N;
        unsigned int c = i / (M * N);

        unsigned int xLoc = x * 8;
        unsigned int yLoc = y * (8 * MR);
        unsigned int cLoc = (c/8) * (8 * MR * N) + c % 8;

        int a;
        fscanf(f, "%4d\n", &a);
        assert((cLoc + yLoc + xLoc) < inTileSize);
        buf[cLoc + yLoc + xLoc] = a;
        if((xLoc == (M-1)) && (((M / MP_S) % 2) == 1)) {
            unsigned int addr = cLoc + yLoc + xLoc + 8;
            assert(addr < inTileSize);
            buf[cLoc + yLoc + xLoc + 8] = 0;
        }
    }

    fclose(f);
}

void writeActsToFile(int* buf, size_t size, const char* const filename) {
    int* reordered = (int*)calloc(1, sizeof(int) * realOutTileSize);

    unsigned int outH = outHeight;
    unsigned int outW = outWidth;

    for(size_t i = 0; i < size; i++) {
        unsigned int c = (i % 8) + (i/(8*outWidth*outHeight)) * 8;
        unsigned int x = (i/8) % outWidth;
        unsigned int y = (i/(8*outWidth)) % outHeight;

        unsigned int ptLoc = c*((M/MP_S)*(N/MP_S)) + y*(M/MP_S) + x;

        // need to remove representation zeros
        if(x < (M/(MP_S))) {
            assert((ptLoc) < realOutTileSize);
            reordered[ptLoc] = buf[i];
        }
    }

    printMatToFile(reordered, realOutTileSize, filename);
    free(reordered);
}

void getActPixelAt(unsigned int x, unsigned int y, unsigned int c, int* acts, int* res) {
    unsigned int xOffset = x * 8;
    unsigned int yOffset = y * MR * 8;
    unsigned int cOffset = c * MR * 8 * N;

    for(unsigned int i = 0; i < 8; i++) {
        res[i] = acts[cOffset + yOffset + xOffset + i];
    }
}

void getActPixelsAt(unsigned int origX, unsigned int origY, unsigned int c, int* acts, int* res) {
    unsigned int i = 0;
    for(unsigned int y = 0; y < MP_W; y++) {
        for(unsigned int x = 0; x < MP_W; x++) {
            getActPixelAt(origX + x, origY + y, c, acts, res + i);
            i += 8;
        }
    }
}

int max(int a, int b) {
    if(a > b) return a; else return b;
}

void avg7x7(int* actsPixels, int64_t* reduced) {
    for(unsigned int i = 0; i < 8; i++) {
        reduced[i] = 0;
        for(unsigned int j = 0; j < MP_W * MP_W; j++) {
            reduced[i] += actsPixels[j*8 + i];
        }

        printf("acc is: %ld\n", reduced[i]);
        reduced[i] = (int64_t)(((double)reduced[i] / (MP_W * MP_W)) + 0.5);
    }
}

void computeOutputPixels(int* acts, int* res) {
    // do 8 MP at once
    int actsPixels [MP_W * MP_W * 8];

    for(unsigned int c = 0; c < C/8; c++) {
        for(unsigned int y = 0; y < (N-(7-1)); y += MP_S) {
            for(unsigned int x = 0; x < (M - (7-1)); x += MP_S) {
                memset(actsPixels, '\0', 8 * MP_W * MP_W);
                getActPixelsAt(x, y, c, acts, actsPixels);

                int64_t reduced[8];
                memset(reduced, '\0', 8 * sizeof(int64_t));
                avg7x7(actsPixels, reduced);

                unsigned int xLoc = (x/MP_S) * 8;
                unsigned int yLoc = (y/MP_S) * (outWidth) * 8;
                unsigned int cLoc = c * outWidth * outHeight * 8;
                for(unsigned int i = 0; i < 8; i++) {
                    //printf("reduced: %d\n", reduced[i]);
                    res[cLoc + yLoc + xLoc +i] = reduced[i];
                }

                if(((x+MP_S) >= (M - (MP_W-1))) && ((M / MP_S) % 2) != 0) {
                    for(unsigned int i = 0; i < 8; i++) {
                        //printf("reduced: %d\n", 0);
                        res[cLoc + yLoc + xLoc + 8 + i] = 0;
                    }
                }
            }
        }
    }
}

// 0 == read generator generated files
// 1 == read reference generated files
int main(int argc, char* argv[]) {
    srand(17);

    int AIn[inTileSize];
    int AOutRef[outTileSize];
    memset(AOutRef, '\0', outTileSize);

    fillAct(AIn, inTileSize);
    printMatToFile(AIn, inTileSize, AIn_FILENAME);

    computeOutputPixels(AIn, AOutRef);

    //printActs(AOutRef, outTileSize);
    printMatToFile(AOutRef, outTileSize, AOutRefReg_FILENAME);
}

