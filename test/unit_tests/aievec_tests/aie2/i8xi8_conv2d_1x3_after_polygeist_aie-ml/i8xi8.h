#ifndef _FILTER_2D_H_
#define _FILTER_2D_H_

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define PARALLEL_FACTOR_8b 16 // Parallelization factor
#define SRS_SHIFT 10          // SRS shift

const int kernel_width = 3;
const int kernel_height = 1;

void conv2d(int8_t *restrict img_in, size_t m1, int8_t *restrict kernel_coeff,
            size_t m2, int32_t *restrict img_out, size_t m3);

#endif
