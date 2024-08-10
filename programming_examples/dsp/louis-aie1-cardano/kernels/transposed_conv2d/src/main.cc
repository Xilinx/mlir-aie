#include <string.h>
#include <stdio.h>

#include "cardano/window/window.h"
#include "cardano/stream/streams.h"
#include <cardano/redefine.h>
#include <cardano/intrinsics.h>

#include "transposed_conv2d.h"
#include "transposed_conv2d_params.h"
#include "test_params.h"

#define S_AIn 0
#define S_WIn 0
#define S_AOutRef 1

alignas(128) int8_t AIn[inTileSize];
alignas(128) uint8_t AOut[outTileSize];
alignas(256) int8_t WIn[weightTileSize];

void fillBuff(int8_t* buff, size_t size, int from) {
    //printf("size: %ld\n", size);
    for(size_t i = 0; i < size; i++) {
        buff[i] = get_ss(from);
        printf("%4d, ", buff[i]);
    }
    printf("\n");
}

int main() {
    printf("Testing M; %d, N; %d, F; %d, Cin; %d, Cout: %d, S: %d\n", M, N, F, Cin, Cout, S);
    //printf("%d\n", outTileSize);
    fillBuff(AIn, inTileSize, S_AIn);
    //printf("Load W...\n");
    fillBuff(WIn, weightTileSize, S_WIn);

    memset(AOut, '\0', outTileSize);

    uint8_t AOutRef[outTileSize];
    fillBuff((int8_t*)AOutRef, outTileSize, S_AOutRef);

    //printf("Init Windows...\n");

    window_internal actInWindow;
    window_internal actOutWindow;
    window_internal weightInWindow;

    size_t vecSizeInA = sizeof(AIn) / sizeof(v16int8); // will load 128 at a time
    window_init(&actInWindow, 1, (v16int8*)AIn, vecSizeInA, vecSizeInA);

    size_t vecSizeOutA = sizeof(AOut) / sizeof(v16int8); // will load 128 at a time
    window_init(&actOutWindow, 1, (v16uint8*)AOut, vecSizeOutA, vecSizeOutA);

    size_t vecSizeInW = sizeof(WIn) / sizeof(v32int8); // will load 256 at a time
    window_init(&weightInWindow, 1, (v32int8*)WIn, vecSizeInW, vecSizeInW);

    //printf("Compute..\n");
    transposed_conv2d_int8_casc_out<M, N, Cin, Cout, F, S, CASC_IN_EN, CASC_OUT_EN>(
                (input_window_int8*)&actInWindow,
                (input_window_int8*)&weightInWindow,
                (input_stream_acc48*)NULL,
                (output_window_uint8*)&actOutWindow);

    //printf("Check...\n");
    for(size_t i = 0; i < outTileSize; i++) {
        printf("%4d, ", AOut[i]);
    }
    printf("\n");

    for(unsigned int i = 0; i < outTileSize; i++) {
        if(AOutRef[i] != AOut[i]) {
            unsigned int x = (i/8) % outWidth;

            if((x < outWidthR)) {
                printf("Failed at %d with ref: %d, computed: %d\n", i, AOutRef[i], AOut[i]);
                printf("FAILED!\n");
                assert(AOutRef[i] == AOut[i]);
                return 1;
            }

        }
    }

    printf("SUCCESS!\n");

    return 0;
}
