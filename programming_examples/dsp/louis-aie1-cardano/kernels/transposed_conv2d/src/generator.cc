#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <cassert>

#include "test_params.h"
#include "transposed_conv2d_params.h"

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
        int sign = (rand() % 2) == 0 ? 1 : -1;
        int value = sign * (rand() % 4);

        //printf("(%4d, %4d, %4d)\n", x, y, c);

        if((c >= Cin) || (x >= M)) {
            //printf("0\n");
            buf[i] = 0;
        } else {
            value = (x + y * MR) & 0x3F;
            if(value > 127) {
                value = -128 - (128 - value);
            }

            int valueR = sign * (rand() % 4);
            buf[i] = valueR;
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
            buf[i] = value;
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
                    const unsigned int Fin) {
    unsigned int i = 0;
    for(unsigned int c = 0; c < (Cin+7)/8; c++) {
        for(unsigned int y = 0; y < Fin; y++) {
            for(unsigned int x = 0; x < Fin; x++) {
                getActPixelAt(origX + x, origY + y, c, acts, res + i);
                i += 8;
            }
        }
    }
}

void getWeightPixelAt(unsigned int w, unsigned int x, unsigned int y, unsigned int c, int* weights, int* res, const unsigned int Fin) {
    unsigned int wOffset = (w/8) * ((Cin+7)/8) * 8 * Fin * Fin * 8
        + ((w%8)/2)*16 + w%2;
    unsigned int xOffset = x * 8 * (8);
    unsigned int yOffset = y * Fin * 8 * 8;
    unsigned int cOffset = c * Fin * Fin * 8 * 8; // TODO support for non-square

    for(unsigned int i = 0; i < 16; i+=2) {
        res[i/2] = weights[wOffset + cOffset + yOffset + xOffset + i];
    }
}

// TODO this function is probably not correct
void getWeightPixelsAt(unsigned int w, int* weights, int* res, const unsigned int Fin) {
    unsigned int i = 0;
    for(unsigned int c = 0; c < (Cin+7)/8; c++) {
        for(unsigned int y = 0; y < Fin; y++) {
            for(unsigned int x = 0; x < Fin; x++) {
                getWeightPixelAt(w, x, y, c, weights, res + i, Fin);
                //printf("(%d, %d, %d, %d): [%d, %d, %d, %d]\n", w, x, y, c, res[i], res[i+1], res[i+2], res[i+3]);
                i += 8;
            }
        }
    }
}

void computeOutputPixels(int* acts, int* weights, int* res, const unsigned int Fin) {
    size_t compute_size = ((Cin+7)/8)* 8 * Fin * Fin;
    int actsPixels [compute_size];
    int weightPixels [compute_size];
    printf("bufSize = %d | %p\n", ((Cin+7)/8)* 8 * Fin * Fin, weightPixels);
    printf("weight size: %d\n", weightTileSize);

    const unsigned int outH = outHeight / 2;
    const unsigned int outW = outWidth / 2;

    for(unsigned int outerW = 0; outerW < (Cout+7)/8; outerW++) {
        for(unsigned int y = 0; y < outH; y+=1) {
            for(unsigned int x = 0; x < outW; x+=1) {
                //printf("iteration (%d, %d, %d)\n", outerW, y, x);
                memset(actsPixels, '\0', compute_size * sizeof(int));
                getActPixelsAt(x, y, acts, actsPixels, Fin);
                for(unsigned int w = 0; w < 8; w++) {
                    memset(weightPixels, '\0', compute_size * sizeof(int));
                    getWeightPixelsAt(outerW * 8 + w, weights, weightPixels, Fin);

                    int acc = 0;
                    for(int i = 0; i < ((Cin+7)/8)* 8 * Fin * Fin; i++) {
                        acc += actsPixels[i] * weightPixels[i];
                    }

                    int tmp = acc;
                    if(acc < 0) {// relu
                        acc = 0;
                    } else if(acc > 255) {// saturation
                        acc = 255;
                    } else {
                        acc = (acc >> SHIFT) & 0xFF;
                    }

                    unsigned int xOffset = x * 8;
                    unsigned int yOffset = y * outW * 8;
                    unsigned int wOffset = outerW * outH * outW * 8;
                    unsigned int index = w + xOffset + yOffset + wOffset;
                    printf("%d : %d | %d\n", index, tmp, weightPixels[0]);
                    res[index] = acc;
                }
            }
        }
    }
}

// The simulator has a different behavior than the transposed kernel
// It has 4 regions and do not store interleaved intermediate results
// This is easier to implement as here we don't care about 
void reorder(int* AOutRef) {
    printf("reorder on %d\n", outTileSize);
    int* AOutRef_final = (int*)malloc(sizeof(int) * outTileSize);

    for(unsigned int i = 0; i < (outTileSize / 8); i++) {
        int from = i / (outTileSize / (4*8));
        int locIndex = i % (outTileSize / (4*8));
        int x = locIndex % MR;
        int y = (locIndex / MR) % N;
        int c = 8 * (locIndex / (MR * N));

        int xLoc = x * 2 + (from % 2);
        int yLoc = y * 2 * outWidth + (from/2) * outWidth;
        int cLoc = c * outWidth * outHeight;
        
        int index = 8 * xLoc + 8 * yLoc + cLoc;

        //printf("%d: ", index);
        for(unsigned int j = 0; j < 8; j++) {
            //printf("%d, ", AOutRef[i*8 + j]);
            assert((index+ j) < outTileSize);
            AOutRef_final[index + j] = AOutRef[i * 8 + j];
        }
        //printf("\n");
    }

    //printf("Final res:\n");
    for(unsigned int i = 0; i < outTileSize; i++) {
        if(i != 0 && ((i%8) == 0)) {
            //printf("\n");
        }

        if(((i/8) % outWidth) == 0 && (i%8) == 0) {
            //printf("\n");
        }
        //printf("%d, ", AOutRef_final[i]);
    }
    //printf("\n");

    memcpy(AOutRef, AOutRef_final, outTileSize * sizeof(int));
    free(AOutRef_final);
}

int main(int argc, char* argv[]) {
    srand(17);

    int AIn[inTileSize];
    int AOutRef[outTileSize];
    int WIn[weightTileSize];

    printf("outW: %d, outH: %d\n", outWidth, outHeight);

    fillAct(AIn, inTileSize);
    fillWeight(WIn, weightTileSize);

    printMatToFile(AIn, inTileSize, AIn_FILENAME);
    printMatToFile(WIn, weightTileSize, WIn_FILENAME);

    //printActs(AIn, inTileSize);
    //printWeights(WIn, weightTileSize);

    printf("%d\n", outTileSize);

    //* We are here using the following transposed2d parameters
    //* Kernel size of 2
    //* Stride of 2 (meaning in the intermediate image we have one zero padding between each pixels)
    //* Padding of 1 around the image
    //* We have the same parameters as described here (example 5), with bigger images
    //* https://makeyourownneuralnetwork.blogspot.com/2020/02/calculating-output-size-of-convolutions.html
    const unsigned int stepA = outTileSize / 4;
    const unsigned int stepW = weightTileSize / 4;
    for(unsigned int i = 0; i < 4; i++) {
        computeOutputPixels(AIn, WIn + stepW * i, AOutRef + stepA * i, 1);
    }
    //printActs(AOutRef, outTileSize);

    reorder(AOutRef);

    printMatToFile(AOutRef, outTileSize, AOutRefReg_FILENAME);
}

