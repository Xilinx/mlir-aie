#ifndef _FILTER_2D_H_
#define _FILTER_2D_H_

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define PARALLEL_FACTOR_8b 16 // Parallelization factor
#define SRS_SHIFT 10          // SRS shift

const int kernel_width = 3;
const int kernel_height = 3;

void conv2d(int8_t *restrict img_in, size_t m3, size_t m4,
            int8_t *restrict kernel_coeff, size_t m, int8_t *restrict img_out,
            size_t m1, size_t m2);
#endif
