#ifndef _FILTER_2D_H_
#define _FILTER_2D_H_

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define PARALLEL_FACTOR_16b 16 // Parallelization factor
#define SRS_SHIFT 10           // SRS shift

const int kernel_width = 3;
const int kernel_height = 3;

void conv2d(int16_t *restrict img_in, size_t m1, size_t m2,
            int16_t *restrict kernel_coeff, size_t m3,
            int16_t *restrict img_out, size_t m4, size_t m5);

void reference(int16_t *restrict img_in, int16_t *restrict kernel_coeff,
               int16_t *restrict img_out, int image_width, int image_height,
               int stride);

#endif
