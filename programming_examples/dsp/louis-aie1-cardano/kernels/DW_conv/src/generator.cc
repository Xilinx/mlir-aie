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
#include "dw_params.h"

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
            buf[i] = (value % 256) / 64;
            //buf[i] = (((i/8)%(MR*N*8)) % 256);
            //if(buf[i] > 127) {
            //    buf[i] = -128 + (buf[i] - 128);
            //}
        }
    }
}

void fillWeight(int* buf, size_t size) {
    for(size_t i = 0; i < size; i++) {
        unsigned int w = (i/4) % 8 + i / (8 * 4 * ((DW_W + 3) / 4) * DW_W);
        unsigned int x = i % 4;
        unsigned int y = i / (4 * 8);

        if(x >= 3) {
            buf[i] = 0;
        } else {
            buf[i] = 1;
        }
    }
}

// TODO not re-tested
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
        if((xLoc == (M-1)) && (((M / DW_S) % 2) == 1)) {
            unsigned int addr = cLoc + yLoc + xLoc + 8;
            assert(addr < inTileSize);
            buf[cLoc + yLoc + xLoc + 8] = 0;
        }
    }

    fclose(f);
}

// TODO not re-tested
void writeActsToFile(int* buf, size_t size, const char* const filename) {
    int* reordered = (int*)calloc(1, sizeof(int) * realOutTileSize);

    unsigned int outH = outHeight;
    unsigned int outW = outWidth;

    for(size_t i = 0; i < size; i++) {
        unsigned int c = (i % 8) + (i/(8*outWidth*outHeight)) * 8;
        unsigned int x = (i/8) % outWidth;
        unsigned int y = (i/(8*outWidth)) % outHeight;

        unsigned int ptLoc = c*((M/DW_S)*(N/DW_S)) + y*(M/DW_S) + x;

        // need to remove representation zeros
        if(x < (M/(DW_S))) {
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
    for(unsigned int y = 0; y < DW_W; y++) {
        for(unsigned int x = 0; x < DW_W; x++) {
            getActPixelAt(origX + x, origY + y, c, acts, res + i);
            i += 8;
        }
    }
}

void getWeightPixelAt(unsigned int x, unsigned int y, unsigned int c, int* weights, int* res, unsigned int i) {
    unsigned int wOffset = (c % 8) * 4 + (c / 8) * 4 * DW_W;
    unsigned int xOffset = x;
    unsigned int yOffset = y * 4 * 8;
    res[i] = weights[wOffset + yOffset + xOffset];
}

void getWeightPixelsAt(unsigned int origC, int* weights, int*res) {
    for(unsigned int c = 0; c < 8; c++) {
        for(unsigned int y = 0; y < DW_W; y++) {
            for(unsigned int x = 0; x < DW_W; x++) {
                getWeightPixelAt(x, y, origC + c, weights, res, x + y * DW_W + c * DW_W * DW_W);
            }
        }
    }
}

void DW3x3(int* actsPixels, int* weightPixels, int64_t* reduced) {
    for(unsigned int i = 0; i < 8; i++) {
        reduced[i] = 0;
        for(unsigned int j = 0; j < DW_W * DW_W; j++) {
            reduced[i] += actsPixels[j*8 + i] * weightPixels[j + i * DW_W * DW_W];
        }
    }
}

void computeOutputPixels(int* acts, int* weights, int* res) {
    // do 8 MP at once
    int actsPixels [DW_W * DW_W * 8];
    int weightPixels [DW_W * DW_W* 8];

    for(unsigned int c = 0; c < C/8; c++) {
        for(unsigned int y = 0; y < (N-(DW_W-1)); y += DW_S) {
            for(unsigned int x = 0; x < (M - (DW_W-1)); x += DW_S) {
                memset(actsPixels, '\0', 8 * DW_W * DW_W);
                getActPixelsAt(x, y, c, acts, actsPixels);

                memset(weightPixels, '\0', 8 * DW_W * DW_W);
                getWeightPixelsAt(c, weights, weightPixels);

                int64_t reduced[8];
                memset(reduced, '\0', 8 * sizeof(int64_t));
                DW3x3(actsPixels, weightPixels, reduced);

                unsigned int xLoc = (x/DW_S) * 8;
                unsigned int yLoc = (y/DW_S) * (outWidth) * 8;
                unsigned int cLoc = c * outWidth * outHeight * 8;

                for(unsigned int i = 0; i < 8; i++) {
                    printf("reduced: %d\n", reduced[i]);
                    if(reduced[i] > 255) {
                        res[cLoc + yLoc + xLoc + i] = 255;
                    } else if(reduced[i] < 0) {
                        res[cLoc + yLoc + xLoc + i] = 0;
                    } else {
                        res[cLoc + yLoc + xLoc + i] = reduced[i];
                    }
                }

                if(((x+DW_S) >= (M - (DW_W-1))) && ((outWidth) % 2) != 0) {
                    for(unsigned int i = 0; i < 8; i++) {
                        printf("reduced: %d\n", 0);
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
    int WIn[weightTileSize];
    int AOutRef[outTileSize];
    memset(AOutRef, '\0', outTileSize);

    fillAct(AIn, inTileSize);
    printMatToFile(AIn, inTileSize, AIn_FILENAME);

    fillWeight(WIn, weightTileSize);
    printMatToFile(WIn, weightTileSize, WIn_FILENAME);

    computeOutputPixels(AIn, WIn, AOutRef);

    //printActs(AOutRef, outTileSize);
    printMatToFile(AOutRef, outTileSize, AOutRefReg_FILENAME);
}

