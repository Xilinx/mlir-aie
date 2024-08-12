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

#include "test_params.h"
#include "conv2d_params.h"

#define FNAME_ACTS "../../reference/activations.csv"
#define FNAME_WEIGHTS "../../reference/filters.csv"
#define FNAME_CONV_RES "../../reference/conv2d.csv"
#define FNAME_RELU_RES "../../reference/relu.csv"

void printActs(int* buf, size_t size) {
    for(size_t i = 0; i < size; i++) {
        if(i != 0 && (i%(8*MR)) == 0) {
            printf("\n\n");
        }

        if(i == (N*MR*8)) {
            printf("==================\n");
        }

        printf("%4d,", buf[i]);
    }
}

void printWeights(int* buf, size_t size) {
    for(size_t i = 0; i < size; i++) {
        if(i != 0 && (i%8) == 0) {
            printf("||");
        }

        if(i == (8*8*F*F)) {
            printf("\n");
        }

        if((i%8) == 0) {
            printf("\n %ld: ", i);
        }

        printf("%4d", buf[i]);
    }
}

void printMatToFile(int* buf, size_t size, const char* const filename) {
    FILE* f = fopen(filename, "w");
    if(f == NULL) {
        fprintf(stderr, "canniot open writen file\n");
        exit(1);
    }

    for(size_t i = 0; i < size; i++) {
        fprintf(f, "%4d\n", buf[i]);
    }

    fflush(f);
    fclose(f);
}

void readActsFromFile(int* buf, size_t size, const char* const filename) {
    FILE* f = fopen(filename, "r");
    if(f == NULL) {
        fprintf(stderr, "Cannot open %s to read\n", filename);
        exit(1);
    }

    for(size_t i = 0; i < size; i++) {
        unsigned int x = i % M;
        unsigned int y = (i / M) % N;
        unsigned int c = i / (M * N);

        unsigned int xLoc = x * 8;
        unsigned int yLoc = y * (8 * MR);
        unsigned int cLoc = (c/8) * (8 * MR * N) + c % 8;

        int a;
        fscanf(f, "%4d\n", &a);
        buf[cLoc + yLoc + xLoc] = a;

        // add some padding to ensure correct data format
        if((xLoc == (M-1)) && ((M%2) == 1)) {
            buf[cLoc + yLoc + xLoc + 8] = 0;
        }
    }

    fclose(f);
}

void writeActsToFile(int* buf, size_t size, const char* const filename) {
    int* reordered = (int*)calloc(1, sizeof(int) * realOutTileSize);
    assert(reordered != NULL);

    printf("%d x %d\n", outWidth, outHeight);

    for(size_t i = 0; i < size; i++) {
        unsigned int c = i % 8 + (i/(8*outWidth*outHeight)) * 8;
        unsigned int x = i/8 % outWidth;
        unsigned int y = i/(8*outWidth) % outHeight;

        unsigned int ptLoc = c*outWidthR*outHeight + y*outWidthR + x;
        //printf("%d\n", ptLoc);

        if(x < (M-(F-1))) {
            assert((ptLoc) < realOutTileSize);
            reordered[ptLoc] = buf[i];
        }
    }

    printMatToFile(reordered, realOutTileSize, filename);
    free(reordered);
}

void readWeightsFromFile(int* buf, size_t size, const char* const filename) {
    FILE* f = fopen(filename, "r");
    for(size_t i = 0; i < size; i++) {
        unsigned int x = i % F;
        unsigned int y = (i / F) % F;
        unsigned int c = (i / (F * F)) % (CinUp);
        unsigned int w = i / (F * F * (CinUp));

        // GlobalweightIndex + LocalWeightIndex + Interleaved offset
        unsigned int wLoc = (w / 8) * CinUp * F * F * 8
            + ((w % 8)/2)*(2*8) + w % 2;
        unsigned int xLoc = x * 8 * 8;
        unsigned int yLoc = y * F * 8 * 8;
        // GlobalCIndex + Local interleaved offset
        unsigned int cLoc = (c/8) * F * F * 8 * 8 + (c % 8) * 2;

        int a;
        fscanf(f, "%4d\n", &a);
        buf[wLoc + cLoc + yLoc + xLoc] = a;
    }
}

void fillAct(int* buf, size_t size) {
    for(size_t i = 0; i < size; i++) {
        unsigned int c = i % 8 + (i/(8*MR*N)) * 8;
        unsigned int x = i/8 % MR;
        unsigned int y = i/(8*MR) % N;

        //int sign = (rand() % 2) == 0 ? 1 : -1;
        int value = (rand() % 256) / 64;

        //printf("(%4d, %4d, %4d)\n", x, y, c);

        if((c >= Cin) || (x >= M)) {
            //printf("0\n");
            buf[i] = 0;
        } else {
            //value = (y * MR + (i % 8)) & 0x3F;
            value = (y * MR + x) & 0x7F;
            //if(value > 127) {
            //    value = -128 - (128 - value);
            //}

            buf[i] = value;
        }
    }
}

void fillWeight(int* buf, size_t size) {
    // C0-7X0Y0
    // then X1 etc to F
    // then Y1 etc to F
    // then C8-15 etc to Cin
    // then w8-15 etc to Cout
    for(size_t i = 0; i < size; i++) {
        unsigned int w = ((i/16)*2 % 8) + i%2 + 8 * (i/(8*8*F*F*(((Cin+7)/8))));
        unsigned int x = (i/(8*8)) % F;
        unsigned int y = (i/(8*8*F)) % F;
        unsigned int cin = ((i/(8*8*F*F))*8 + ((i/2)) % 8) % (((Cin+7)/8)*8);

        //printf("(%4d, %4d, %4d, %4d)\n", x, y, cin, w);

        int sign = (rand() % 2) == 0 ? 1 : -1;
        int value = sign * (rand() % 8);
        if(cin >= Cin || w >= Cout) {
            //printf("zero: (c:%d, i:%ld)\n", cin, i);
            buf[i] = 0;
        } else {
            //buf[i] = value;
            buf[i] = 1;
        }
    }
}

void getActPixelAt(unsigned int x, unsigned int y, unsigned int c, int* acts, int* res) {
    unsigned int xOffset = x * 8;
    unsigned int yOffset = y * MR * 8;
    unsigned int cOffset = c * MR * 8 * N;

    for(unsigned int i = 0; i < 8; i++) {
        res[i] = acts[cOffset + yOffset + xOffset + i];
    }
}

void getActPixelsAt(unsigned int origX, unsigned int origY, int* acts, int* res,
                    unsigned int fromX, unsigned int fromY, unsigned int endX, unsigned int endY) {
    unsigned int i = 0;
    for(unsigned int c = 0; c < (Cin+7)/8; c++) {
        for(unsigned int y = 0; y < (endY - fromY); y++) {
            for(unsigned int x = 0; x < (endX - fromX); x++) {
                getActPixelAt(origX + x, origY + y, c, acts, res + i);
                i += 8;
            }
        }
    }
}

void getWeightPixelAt(unsigned int w, unsigned int x, unsigned int y, unsigned int c, int* weights, int* res) {
    unsigned int wOffset = (w/8) * ((Cin+7)/8) * 8 * F * F * 8
        + ((w%8)/2)*16 + w%2;
    unsigned int xOffset = x * 8 * (8);
    unsigned int yOffset = y * F * 8 * 8;
    unsigned int cOffset = c * F * F * 8 * 8;

    for(unsigned int i = 0; i < 16; i+=2) {
        res[i/2] = weights[wOffset + cOffset + yOffset + xOffset + i];
    }
}

void getWeightPixelsAt(unsigned int w, int* weights, int* res,
                       unsigned int fromX, unsigned int fromY,
                       unsigned int endX, unsigned int endY) {
    unsigned int i = 0;
    for(unsigned int c = 0; c < (Cin+7)/8; c++) {
        for(unsigned int y = fromY; y < endY; y++) {
            for(unsigned int x = fromX; x < endX; x++) {
                getWeightPixelAt(w, x, y, c, weights, res + i);
                //printf("(%d, %d, %d, %d): [%d, %d, %d, %d]\n", w, x, y, c, res[i], res[i+1], res[i+2], res[i+3]);
                i += 8;
            }
        }
    }
}

unsigned int max(int a, int b) {
    return (a > b) ? a : b;
}

void computeOutputPixels(int* acts, int* weights, int* res) {
    int actsPixels [((Cin+7)/8)* 8 * F * F];
    int weightPixels [((Cin+7)/8)* 8 * F * F];

    printf("weight size: %d\n", weightTileSize);
    printf("acts size: %d\n", ((Cin+7)/8)* 8 * F * F);

    for(unsigned int outerW = 0; outerW < (Cout+7)/8; outerW++) {
            for(unsigned int y = 0; y < outHeight; y+=1) {
                for(unsigned int x = 0; x < outWidth; x+=1) {
                //printf("iteration (%d, %d, %d)\n", outerW, y, x);
                memset(actsPixels, 0, ((Cin+7)/8)* 8 * F * F * sizeof(int));

                unsigned int inX = max(x * S - (int)P, 0);
                unsigned int inY = max(y * S - (int)P, 0);

                printf("inx: %d, inY: %d\n", inX, inY);

                unsigned int outX = x;
                unsigned int outY = y;

                unsigned int fromX = ((int)(P - outX) > 0) ? (P - outX) : 0;
                unsigned int fromY = (int)(P - outY) > 0 ? (P - outY) : 0;
                unsigned int endX = (int)(MR - outX) > P ? F : (F - (MR - outX));
                unsigned int endY = (int)(N - outY) > P ? F : (F - (N - outY));

                getActPixelsAt(inX, inY, acts, actsPixels, fromX, fromY, endX, endY);
                for(unsigned int w = 0; w < 8; w++) {
                    memset(weightPixels, 0, ((Cin+7)/8)* 8 * F * F* sizeof(int));
                    getWeightPixelsAt(outerW * 8 + w, weights, weightPixels,
                                    fromX, fromY, endX, endY);

                    int acc = 0;
                    for(int i = 0; i < ((Cin+7)/8)* 8 * F * F; i++) {
                        acc += actsPixels[i] * weightPixels[i];
                        //printf("%d: acts: %d\n", i, actsPixels[i]);
                    }

                    int tmp = acc;
                    if(acc < 0) {// relu
                        acc = 0;
                        //acc = (acc >> 0) & 0xFF;
                    } else if(acc > 255) {// saturation
                        acc = 255;
                        //acc = (acc >> 0) & 0xFF;
                    } else {
                        //printf("%d\n", acc >> SHIFT);
                        acc = (acc >> SHIFT) & 0xFF;
                    }

                    unsigned int xOffset = x * 8;
                    unsigned int yOffset = y * outWidth * 8;
                    unsigned int wOffset = outerW * outHeight * outWidth * 8;
                    unsigned int index = w + xOffset + yOffset + wOffset;
                    printf("%d : %d | %d(%d, %d, %d, %d)\n", index, tmp, weightPixels[0], fromX, fromY, endX, endY);
                    res[index] = acc;
                }
            }
        }
    }
}

void fillBuff(int* buff, size_t size, char* fname) {
    FILE* f = fopen(fname, "r");

    for(size_t i = 0; i < size; i++) {
        fscanf(f, "%d\n", &buff[i]);

        //printf("%4d, ", buff[i]);
    }
    printf("\n");
}

// 0 == read generator generated files
// 1 == read reference generated files
int main(int argc, char* argv[]) {
    srand(17);

    int AIn[inTileSize];
    int AOutRef[outTileSize];
    int WIn[weightTileSize];

    printf("outW: %d, outH: %d, inTileSize: %d\n", outWidth, outHeight, inTileSize);

    char* fnameAIn = "../data/AIn.txt";
    char* fnameWIn = "../data/WIn.txt";

    //fillBuff(AIn, inTileSize, fnameAIn);
    //fillBuff(WIn, inTileSize, fnameWIn);

    //return 0;

    fillAct(AIn, inTileSize);
    fillWeight(WIn, weightTileSize);

    //printMatToFile(AIn, inTileSize, AIn_FILENAME);
    //printMatToFile(WIn, weightTileSize, WIn_FILENAME);

    //printActs(AIn, inTileSize);
    //printWeights(WIn, weightTileSize);

    printf("%d\n", outTileSize);

    computeOutputPixels(AIn, WIn, AOutRef);
    //printActs(AOutRef, outTileSize);

    //printMatToFile(AOutRef, outTileSize, AOutRefReg_FILENAME);
}

