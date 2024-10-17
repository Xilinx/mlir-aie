#ifndef _FILTER_2D_H_
#define _FILTER_2D_H_

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define PARALLEL_FACTOR_FLOAT 8 // Parallelization factor
#define SRS_SHIFT 10            // SRS shift

const int kernel_width = 3;
const int kernel_height = 3;

void reference(float *restrict img_in, float *restrict kernel_coeff,
               float *restrict img_out, int image_width, int image_height,
               int stride);

void conv2d(float *restrict img_in, size_t m1, size_t m2,
            float *restrict kernel_coeff, size_t m3, float *restrict img_out,
            size_t m4, size_t m5);

#endif
